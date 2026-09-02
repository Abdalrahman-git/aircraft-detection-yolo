"""YOLOv9 fine-tuning track.

The from-scratch detector in the rest of this package is the baseline; this
subpackage fine-tunes a pretrained YOLOv9 so the two can be compared on equal
terms. It is optional and imports `ultralytics` lazily, so the core package
keeps its small dependency set:

    pip install -e ".[yolov9]"
"""

__all__ = ["export_for_ultralytics", "write_data_yaml"]


def __getattr__(name: str):
    if name in __all__:
        from . import export

        return getattr(export, name)
    raise AttributeError(name)
