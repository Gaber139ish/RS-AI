import os
from copy import deepcopy
from typing import Any, Dict


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "on", "y"}


def apply_low_memory_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of config with low-memory overrides if enabled.

    Enable via env RS_LOW_MEM=1 or config.profiles.low_memory.enabled=true
    """
    cfg = deepcopy(config)
    lm = (cfg.get("profiles", {}).get("low_memory", {}) if isinstance(cfg.get("profiles"), dict) else {})
    enabled = _truthy(os.getenv("RS_LOW_MEM")) or _truthy(lm.get("enabled", False))
    if not enabled:
        return cfg

    # Defaults for Chromebook-class constraints; allow overriding in config
    overrides = {
        "filepaths": {
            "sponge_size": lm.get("sponge_size", [7, 7, 7]),
        },
        "memory": {
            "backend": lm.get("memory_backend", "hffs"),
        },
        "training": {
            "batch_size": int(lm.get("batch_size", 2)),
            "max_steps": int(lm.get("max_steps", 100)),
            "replay_capacity": int(lm.get("replay_capacity", 200)),
            "ssm_hidden": int(lm.get("ssm_hidden", 64)),
            "clip_norm": float(lm.get("clip_norm", 0.5)),
            "enable_world_model": bool(lm.get("enable_world_model", False)),
            "world_lr": float(lm.get("world_lr", 5e-4)),
        },
        "compression": {
            "enabled": bool(lm.get("compression_enabled", False)),
            "rank": int(lm.get("compression_rank", 4)),
        },
        "dashboard": {
            "enabled": bool(lm.get("dashboard_enabled", False)),
        },
        "auto": {
            "explore_prob": float(lm.get("explore_prob", 0.02)),
        },
        "orchestrator": {
            "runtime_seconds": int(lm.get("runtime_seconds", 6)),
        },
    }

    # Merge overrides into cfg (shallow nested merge for known sections)
    for section, vals in overrides.items():
        sec = cfg.get(section, {}) if isinstance(cfg.get(section), dict) else {}
        sec.update(vals)
        cfg[section] = sec

    # Optionally drop heavy modules, e.g., world_model
    if not cfg.get("training", {}).get("enable_world_model", True):
        mods = cfg.get("modules", {})
        if isinstance(mods, dict) and "world_model" in mods:
            mods = {k: v for k, v in mods.items() if k != "world_model"}
            cfg["modules"] = mods

    return cfg