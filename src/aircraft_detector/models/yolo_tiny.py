"""A compact YOLOv5-style single-class detector, written from scratch.

Architecture::

    stem (s2) -> CSP x4 (s2 each) -> SPPF -> 1x1 head

Total stride is 32, so a 640x640 input produces a 20x20 prediction grid. Each
cell predicts one box: ``(tx, ty, w, h, objectness)`` where ``tx, ty`` are
offsets within the cell and ``w, h`` are normalised to the whole image.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STRIDE = 32  # stem + 4 CSP stages, each stride 2


class ConvBNSiLU(nn.Module):
    """Conv -> BatchNorm -> SiLU, the base unit of every YOLOv5 block.

    ``padding`` defaults to ``kernel // 2``, which halves the resolution exactly
    for odd kernels at stride 2. Even kernels need ``(kernel - stride) // 2``
    instead, so they must pass it explicitly - see the stem in :class:`YOLOTiny`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.03, eps=1e-3),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSPBlock(nn.Module):
    """Cross Stage Partial block.

    Downsamples, then splits into a convolutional path that learns features and
    a shortcut path that preserves the input signal. Concatenating the two gives
    richer gradient flow than a plain conv stack at a lower cost.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        mid = out_ch // 2
        self.down = ConvBNSiLU(in_ch, out_ch, 3, stride)
        self.path1 = nn.Sequential(
            ConvBNSiLU(out_ch, mid, 1),
            ConvBNSiLU(mid, mid, 3),
            ConvBNSiLU(mid, mid, 3),
        )
        self.path2 = ConvBNSiLU(out_ch, mid, 1)
        self.merge = ConvBNSiLU(mid * 2, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.merge(torch.cat([self.path1(x), self.path2(x)], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Three chained 5x5 max-pools give effective receptive fields of 5, 9 and 13
    and are concatenated, so the head sees aircraft at several apparent sizes
    without extra convolution cost.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = channels // 2
        self.cv1 = ConvBNSiLU(channels, mid, 1)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.cv2 = ConvBNSiLU(mid * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class YOLOTiny(nn.Module):
    """Single-class detector. Output: ``[B, S, S, 5]`` with every value in (0, 1)."""

    def __init__(self, width: tuple[int, ...] = (16, 32, 64, 128, 256)) -> None:
        super().__init__()
        stem_ch, *stage_ch = width
        # padding=2, not kernel//2=3. With padding 3 a 6x6/stride-2 conv emits
        # `in/2 + 1` rows, so 640px produced a 21x21 grid whose cells no longer
        # line up with the [j/S, (j+1)/S] image spans the decoder assumes.
        self.stem = ConvBNSiLU(3, stem_ch, kernel=6, stride=2, padding=2)
        channels = [stem_ch, *stage_ch]
        self.stages = nn.Sequential(
            *[CSPBlock(channels[i], channels[i + 1]) for i in range(len(stage_ch))]
        )
        self.sppf = SPPF(channels[-1])
        self.head = nn.Conv2d(channels[-1], 5, kernel_size=1)

        # Bias objectness towards "empty" at init. Over 99% of cells are
        # background, so starting near p=0.01 stops the first epochs from being
        # dominated by the model unlearning a uniform "object everywhere" prior.
        with torch.no_grad():
            self.head.bias[4].fill_(-4.6)  # sigmoid(-4.6) ~= 0.01

    @staticmethod
    def grid_size(image_size: int) -> int:
        if image_size % STRIDE != 0:
            raise ValueError(f"image_size {image_size} must be a multiple of stride {STRIDE}")
        return image_size // STRIDE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.sppf(x)
        x = torch.sigmoid(self.head(x))
        return x.permute(0, 2, 3, 1).contiguous()  # [B, 5, S, S] -> [B, S, S, 5]
