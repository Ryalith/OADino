from pathlib import Path

_DATA_FOLDER = Path.home() / ".cache" / "oadino"


def get_data_path() -> Path:
    return _DATA_FOLDER


def set_data_path(path: Path):
    global _DATA_FOLDER
    _DATA_FOLDER = path
