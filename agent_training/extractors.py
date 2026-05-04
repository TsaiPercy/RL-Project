"""Custom feature extractor for BabyAI dict observations.

Per SPEC §5.4 [impl-updated]: ``BabyAIDictExtractor`` consumes the dict
produced by ``MissionTokenizer`` (which exposes the symbolic image as
``int64`` so SB3 does not RGB-normalise it) and fuses three modalities
into a single feature vector for SB3's ``MultiInputPolicy``::

    {"image":     (B, H, W, 3),  # symbolic ids: object_type / color / state
     "direction": (B, 1),         # int64 in {0,1,2,3} (cast to float by SB3)
     "mission":   (B, L)}         # int64 token ids (cast to float by SB3)

Pipeline:

    image (3 symbolic channels)
      -> Embedding × 3 (object/color/state) -> concat per cell
      -> CNN (k=2 ×3 on cell-embedding map)  -> Linear -> img_feat
    mission   -> Embedding -> masked mean-pool -> Linear -> text_feat
    direction -> Embedding(4)                                -> dir_feat

    [img_feat || text_feat || dir_feat] -> Linear -> features_dim

Why per-channel embeddings (not a CNN-on-RGB): BabyAI's image is a
symbolic encoding with values in 0-10, NOT pixel intensities.  Treating
it as a uint8 RGB image causes SB3 to divide by 255, crushing every
value to ~0 and starving the CNN of spatial signal — the policy then
collapses to mission-conditioned random walk.  This module follows the
BabyAI paper's standard recipe (Chevalier-Boisvert et al., 2018):
embed each symbolic channel independently, then run a small CNN over
the resulting cell-embedding feature map.

The class is defined at module level (not inside a trainer method) so
``cloudpickle`` can resolve it on ``PPO.load(...)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, reduce
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_training.wrappers import (
    NUM_IMAGE_COLORS,
    NUM_OBJECT_STATES,
    NUM_OBJECT_TYPES,
)

# CNN block on the embedded cell map.  The (16, 32, 64) channel widths
# and k=2 kernel size are the same as the X-1 design that worked on the
# 7x7 BabyAI obs (NatureCNN's k=8 is incompatible with 7x7).
CONV1_OUT: int = 16
CONV2_OUT: int = 32
CONV3_OUT: int = 64
KERNEL_SIZE: int = 2
IMG_PROJ_DIM: int = 128
TEXT_PROJ_DIM: int = 64
NUM_DIRECTIONS: int = 4  # MiniGrid agent direction is in {0, 1, 2, 3}

# Image-channel ordering matches BabyAI's ``OBJECT_TYPE_IDX/COLOR_IDX/STATE_IDX``
# layout exposed by ``ImgObsWrapper`` and the raw env: cell = (type, color, state).
IMG_CHANNEL_OBJECT: int = 0
IMG_CHANNEL_COLOR: int = 1
IMG_CHANNEL_STATE: int = 2


class BabyAIDictExtractor(BaseFeaturesExtractor):
    """Fuse symbolic BabyAI image + mission tokens + direction into features.

    SB3's pipeline before this extractor:
      * image: ``Box(int64, (H,W,3))`` — NOT detected as an image space
        (dtype is int64, not uint8), so neither ``VecTransposeImage`` nor
        the ``/255`` normalisation fire. ``preprocess_obs`` casts to float.
      * mission/direction: ``Box(int64, ...)`` cast to float by
        ``preprocess_obs``; we cast back to ``long()`` for embedding lookup.

    Args:
        observation_space: The Dict space produced by ``MissionTokenizer``.
        features_dim: Output dimension of the fused feature vector.
        vocab_size: Mission vocabulary size (``len(BABYAI_VOCAB)``).
        text_embed_dim: Embedding width for mission tokens.
        dir_embed_dim: Embedding width for the agent direction.
        obj_embed_dim: Embedding width for the image's object_type channel.
        color_embed_dim: Embedding width for the image's color channel.
        state_embed_dim: Embedding width for the image's state channel.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int,
        vocab_size: int,
        text_embed_dim: int,
        dir_embed_dim: int,
        obj_embed_dim: int,
        color_embed_dim: int,
        state_embed_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # --- image (symbolic) branch ---
        self.obj_emb = nn.Embedding(NUM_OBJECT_TYPES, obj_embed_dim)
        self.color_emb_img = nn.Embedding(NUM_IMAGE_COLORS, color_embed_dim)
        self.state_emb = nn.Embedding(NUM_OBJECT_STATES, state_embed_dim)

        cell_emb_dim: int = obj_embed_dim + color_embed_dim + state_embed_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(cell_emb_dim, CONV1_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Conv2d(CONV1_OUT, CONV2_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Conv2d(CONV2_OUT, CONV3_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Auto-detect the flattened CNN output dim with a sample tensor
        # that mirrors the runtime cell-embedding layout.
        image_space = observation_space["image"]
        h: int = int(image_space.shape[0])
        w: int = int(image_space.shape[1])
        with torch.no_grad():
            dummy_cells = torch.zeros(
                1, cell_emb_dim, h, w, dtype=torch.float32
            )  # (1, cell_emb_dim, H, W)
            n_flat: int = self.cnn(dummy_cells).shape[1]  # (1, n_flat)

        self.img_proj = nn.Sequential(
            nn.Linear(n_flat, IMG_PROJ_DIM),
            nn.ReLU(),
        )

        # --- mission text branch ---
        self.text_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embed_dim,
            padding_idx=0,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, TEXT_PROJ_DIM),
            nn.ReLU(),
        )

        # --- direction branch ---
        self.dir_emb = nn.Embedding(
            num_embeddings=NUM_DIRECTIONS,
            embedding_dim=dir_embed_dim,
        )

        # --- fusion ---
        fused_dim: int = IMG_PROJ_DIM + TEXT_PROJ_DIM + dir_embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute fused features from a batched dict observation.

        Args:
            obs: Dict with keys
              ``image``     (B, H, W, 3) float (cast int64 by SB3),
              ``direction`` (B, 1)        float,
              ``mission``   (B, L)        float.

        Returns:
            Tensor of shape (B, features_dim).
        """
        symbolic = obs["image"].long()  # (B, H, W, 3)
        obj_e = self.obj_emb(symbolic[..., IMG_CHANNEL_OBJECT])  # (B, H, W, obj_e)
        col_e = self.color_emb_img(symbolic[..., IMG_CHANNEL_COLOR])  # (B, H, W, col_e)
        st_e = self.state_emb(symbolic[..., IMG_CHANNEL_STATE])  # (B, H, W, st_e)
        cells_hwc = torch.cat([obj_e, col_e, st_e], dim=-1)  # (B, H, W, cell_emb_dim)
        cells = rearrange(cells_hwc, "b h w c -> b c h w")  # (B, cell_emb_dim, H, W)
        cnn_out = self.cnn(cells)  # (B, n_flat)
        img_feat = self.img_proj(cnn_out)  # (B, IMG_PROJ_DIM)

        tokens = obs["mission"].long()  # (B, L)
        token_embeds = self.text_emb(tokens)  # (B, L, text_embed_dim)
        mask = rearrange((tokens != 0).float(), "b l -> b l 1")  # (B, L, 1)
        masked = token_embeds * mask  # (B, L, text_embed_dim)
        denom = reduce(mask, "b l 1 -> b 1", "sum").clamp(min=1.0)  # (B, 1)
        pooled = reduce(masked, "b l e -> b e", "sum") / denom  # (B, text_embed_dim)
        text_feat = self.text_proj(pooled)  # (B, TEXT_PROJ_DIM)

        dir_idx = rearrange(obs["direction"].long(), "b 1 -> b")  # (B,)
        dir_feat = self.dir_emb(dir_idx)  # (B, dir_embed_dim)

        fused = torch.cat([img_feat, text_feat, dir_feat], dim=-1)  # (B, fused_dim)
        return self.fuse(fused)  # (B, features_dim)
