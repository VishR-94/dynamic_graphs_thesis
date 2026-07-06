from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Expected a config file, got: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config