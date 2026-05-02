"""Custom feature extractor for BabyAI dict observations.

Per SPEC §5.4 [impl-updated]: ``BabyAIDictExtractor`` consumes the dict
produced by ``MissionTokenizer`` and fuses three modalities into a single
feature vector for SB3's ``MultiInputPolicy``::

    {"image": (B, C, H, W),    # uint8 image, transposed + normalised by SB3
     "direction": (B, 1),       # int64 in {0,1,2,3} (cast to float by SB3)
     "mission":  (B, L)}        # int64 token ids (cast to float by SB3)

Pipeline (all dims from config; only the proven k=2 conv stack and the
small projection widths are inline architectural constants):

    image     -> Conv k=2 ×3 (C->16->32->64) -> Linear -> img_feat (B, 128)
    mission   -> Embedding(vocab) -> mean-pool with PAD mask
                 -> Linear         -> text_feat (B, 64)
    direction -> Embedding(4)      -> dir_feat  (B, dir_embed_dim)

    [img_feat || text_feat || dir_feat] -> Linear -> features_dim

The class is defined at module level (not inside a trainer method) so that
``cloudpickle`` can resolve it on ``PPO.load(...)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, reduce
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Image branch: a 3-conv k=2 stack that fits any partial obs >= 4x4.
# This is the same architecture that AT-4 / X-1 validated to replace SB3's
# NatureCNN (whose 8x8 first kernel exceeds the 7x7 BabyAI obs).  The
# channel widths (16/32/64) and projection widths below are documented
# architectural constants — not researcher-tunable hyperparameters.
CONV1_OUT: int = 16
CONV2_OUT: int = 32
CONV3_OUT: int = 64
KERNEL_SIZE: int = 2
IMG_PROJ_DIM: int = 128
TEXT_PROJ_DIM: int = 64
NUM_DIRECTIONS: int = 4  # MiniGrid agent direction is in {0, 1, 2, 3}


class BabyAIDictExtractor(BaseFeaturesExtractor):
    """Fuse BabyAI image, mission tokens, and direction into a feature vector.

    SB3 invokes us with already-preprocessed inputs:
      * image: ``Box(uint8, (H,W,C))`` is auto-transposed by
        ``VecTransposeImage`` to ``(C,H,W)`` at the env level and
        normalised to ``[0,1]`` by SB3's ``preprocess_obs``.
      * mission/direction: ``Box(int64, ...)`` is cast to float by
        ``preprocess_obs``; we cast back to ``long()`` for embedding lookup.

    Args:
        observation_space: The Dict space produced by ``MissionTokenizer``.
        features_dim: Output dimension of the fused feature vector.
        vocab_size: Size of the mission vocabulary (``len(BABYAI_VOCAB)``).
        text_embed_dim: Embedding width for mission tokens.
        dir_embed_dim: Embedding width for the agent direction.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int,
        vocab_size: int,
        text_embed_dim: int,
        dir_embed_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Detect channel position once: SB3's ``VecTransposeImage`` rewrites
        # the policy's image space from HWC to CHW before the extractor is
        # built, but only when the original env produced HWC.  We support
        # both so this extractor works whether or not the auto-transpose
        # fired.  ``self._image_channels_first`` is consulted again inside
        # ``forward`` because the runtime tensor layout always matches
        # ``observation_space["image"]``.
        image_space = observation_space["image"]
        self._image_channels_first: bool = is_image_space_channels_first(image_space)
        if self._image_channels_first:
            n_channels: int = int(image_space.shape[0])  # (C, H, W) -> C
        else:
            n_channels = int(image_space.shape[-1])  # (H, W, C) -> C

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, CONV1_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Conv2d(CONV1_OUT, CONV2_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Conv2d(CONV2_OUT, CONV3_OUT, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Auto-detect the flattened CNN output dim with a sample tensor that
        # mirrors the runtime layout.  ``image_space.sample()`` follows the
        # space's stored layout, so we only rearrange if the space is HWC.
        with torch.no_grad():
            sample = torch.as_tensor(image_space.sample()[None]).float()  # (1, *space.shape)
            if not self._image_channels_first:
                sample = rearrange(sample, "b h w c -> b c h w")  # (1, C, H, W)
            n_flat: int = self.cnn(sample).shape[1]  # (1, n_flat)

        self.img_proj = nn.Sequential(
            nn.Linear(n_flat, IMG_PROJ_DIM),
            nn.ReLU(),
        )

        self.text_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embed_dim,
            padding_idx=0,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, TEXT_PROJ_DIM),
            nn.ReLU(),
        )

        self.dir_emb = nn.Embedding(
            num_embeddings=NUM_DIRECTIONS,
            embedding_dim=dir_embed_dim,
        )

        fused_dim: int = IMG_PROJ_DIM + TEXT_PROJ_DIM + dir_embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute fused features from a batched dict observation.

        Args:
            obs: Dict with keys ``image`` (B, C, H, W) float in [0,1],
                ``direction`` (B, 1) float (cast int), ``mission``
                (B, L) float (cast int).

        Returns:
            Tensor of shape (B, features_dim).
        """
        image = obs["image"]  # (B, *image_space.shape) float in [0, 1]
        if not self._image_channels_first:
            image = rearrange(image, "b h w c -> b c h w")  # (B, C, H, W)
        cnn_out = self.cnn(image)  # (B, n_flat)
        img_feat = self.img_proj(cnn_out)  # (B, IMG_PROJ_DIM)

        tokens = obs["mission"].long()  # (B, L)
        token_embeds = self.text_emb(tokens)  # (B, L, text_embed_dim)
        # Mean-pool over non-PAD tokens.  Shapes are commented end-to-end
        # so the pooling math is auditable.
        mask = rearrange((tokens != 0).float(), "b l -> b l 1")  # (B, L, 1)
        masked = token_embeds * mask  # (B, L, text_embed_dim)
        denom = reduce(mask, "b l 1 -> b 1", "sum").clamp(min=1.0)  # (B, 1)
        pooled = reduce(masked, "b l e -> b e", "sum") / denom  # (B, text_embed_dim)
        text_feat = self.text_proj(pooled)  # (B, TEXT_PROJ_DIM)

        dir_idx = rearrange(obs["direction"].long(), "b 1 -> b")  # (B,)
        dir_feat = self.dir_emb(dir_idx)  # (B, dir_embed_dim)

        fused = torch.cat([img_feat, text_feat, dir_feat], dim=-1)  # (B, fused_dim)
        return self.fuse(fused)  # (B, features_dim)
