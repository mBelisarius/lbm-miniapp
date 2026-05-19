"""Helpers shared by every analysis module.

The module concentrates the constants, the path resolution and the file
input/output used by the campaign runner, by the report builder and by the
standalone analyses, so no other module needs to know how the repository is
laid out on disk.
"""
import json
import os
import signal
import yaml

from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DEFAULT_CAMPAIGN = ANALYSIS_DIR / 'campaign.yaml'

CS2 = 1.0 / 3.0


class CampaignInterrupt(Exception):
    """Raised when the campaign receives an interruption signal."""


def resolve_path(path):
    """Resolves a configured path against the repository root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, payload):
    with open(path, 'w') as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)


def write_json(path, payload):
    """Writes JSON atomically so an interruption never truncates the ledger."""
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def install_signal_handlers():
    """Turns interruption signals into CampaignInterrupt exceptions."""
    def handler(signum, _frame):
        raise CampaignInterrupt(f"interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (AttributeError, OSError, ValueError):
            pass


def select_backend(interactive=False):
    """Selects the matplotlib backend before pyplot is imported anywhere."""
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')
