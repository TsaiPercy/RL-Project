"""Set all ML-related cache dirs to project-local paths.

Import this module at the very top of entry-point scripts,
before any ML library (transformers, torch, wandb) is imported.

Usage:
    from shared.env_setup import setup_project_cache
    setup_project_cache()          # auto-detects project root
"""

from __future__ import annotations

import os
from pathlib import Path


def setup_project_cache(project_root: str | None = None) -> Path:
    """Configure all ML cache dirs to live under <project_root>/cache/.

    Args:
        project_root: Absolute path to the project root. Defaults to the
            parent directory of this file's parent (i.e. the repo root).

    Returns:
        The cache root Path that was configured.
    """
    if project_root is None:
        root = Path(__file__).parent.parent.resolve()
    else:
        root = Path(project_root).resolve()

    cache_root = root / "cache"
    hf_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    wandb_dir = cache_root / "wandb"

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_dir))

    for d in [hf_home, torch_home, wandb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return cache_root
