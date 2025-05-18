import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    cfg_path = Path(path)
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)
