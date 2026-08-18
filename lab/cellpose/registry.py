"""Resolve neuron / astrocyte Cellpose model paths.

Weights live in the repo-root `models/` directory (gitignored). Training a
custom astrocyte model should drop files there and point CELLPOSE['models']
at them; runners can then pass the path into suite2p ops['pretrained_model'].
"""

from pathlib import Path

from lab.configs.defaults import CELLPOSE

_REPO_ROOT = Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return _REPO_ROOT / "models"


def model_path(cell_type: str):
    """Return a Path to the configured model for 'neuron' or 'astrocyte', or None."""
    name = CELLPOSE["models"].get(cell_type)
    if not name:
        return None
    path = Path(name)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def model_for_channel(channel_name: str):
    """Return (cell_type, model_path) for a channel folder name, or (None, None)."""
    cell_type = CELLPOSE["channel_models"].get(channel_name)
    if not cell_type:
        return None, None
    return cell_type, model_path(cell_type)
