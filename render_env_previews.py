"""Render PNG previews for every curriculum environment.

Reads the curriculum list from config/default.yaml and saves N screenshots
per environment (each from a different random reset) to an output directory.

Output layout:
  {output_dir}/
    01_BabyAI-GoToObjS4-v0/
      01.png ... N.png
    02_BabyAI-GoToObj-v0/
      01.png ... N.png
    ...

Usage:
  python render_env_previews.py
  python render_env_previews.py --n 5 --out my_previews
  python render_env_previews.py --config config/default.yaml --out env_previews
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_OUTPUT_DIR = "env_previews"
DEFAULT_N_IMAGES = 10


def render_previews(
    curriculum: list[dict],
    output_dir: Path,
    n_images: int,
    arg: argparse.Namespace = argparse.Namespace(),
) -> None:
    """Render and save PNG previews for each curriculum environment.

    Args:
        curriculum: List of dicts with "env" key (from config).
        output_dir: Root directory to write previews into.
        n_images: Number of screenshots per environment.
    """
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers BabyAI envs
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    

    if not arg.specific_env:
        for level_idx, level_cfg in enumerate(curriculum):
            env_id: str = level_cfg["env"]
            level_dir = output_dir / f"{level_idx + 1:02d}_{env_id}"
            level_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{level_idx + 1}/{len(curriculum)}] {env_id}")

            env = gym.make(env_id, render_mode="rgb_array")
            for i in range(n_images):
                env.reset(seed=i)
                rgb = env.render()  # (H, W, 3) uint8
                out_path = level_dir / f"{i + 1:02d}.png"
                Image.fromarray(rgb).save(out_path)
                print(f"  saved {out_path}")
            env.close()
    else:
        # Only render previews for the specified env ID (overrides curriculum list).
        env_id = arg.specific_env
        level_dir = output_dir / f"{env_id}"
        level_dir.mkdir(parents=True, exist_ok=True)

        print(f"Rendering previews for specific env: {env_id}")

        env = gym.make(env_id, render_mode="rgb_array")
        for i in range(n_images):
            env.reset(seed=i)
            rgb = env.render()  # (H, W, 3) uint8
            out_path = level_dir / f"{i + 1:02d}.png"
            Image.fromarray(rgb).save(out_path)
            print(f"  saved {out_path}")
        env.close()
    print(f"\nDone. Previews written to: {output_dir}/")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config (default: config/default.yaml).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: env_previews/).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_IMAGES,
        help="Number of screenshots per environment (default: 10).",
    )
    parser.add_argument(
        "--specific-env",
        type=str,
        default=None,
        help="Only render previews for this specific env ID (overrides curriculum list).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    curriculum: list[dict] = config["agent_training"]["curriculum"]
    
    print(f"Rendering {args.n} previews × {len(curriculum)} environments → {args.out}/\n")

    render_previews(
        curriculum=curriculum,
        output_dir=Path(args.out),
        n_images=args.n,
        arg=args
    )


if __name__ == "__main__":
    main()
