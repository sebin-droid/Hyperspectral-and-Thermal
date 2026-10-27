from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
LOCAL_CONFIG = DEFAULT_CONFIG.with_name("local.yaml")    # per-machine paths, not in git


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    """Default config, overridden by configs/local.yaml (if present) and then by `path` (if given)."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for p in (LOCAL_CONFIG if LOCAL_CONFIG.exists() else None, path):
        if p is not None:
            cfg = _merge(cfg, yaml.safe_load(Path(p).read_text()) or {})
    return cfg
