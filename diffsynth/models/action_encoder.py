import math

import torch
import torch.nn as nn


# ============================================================================
# ActionFFNEncoder
# ============================================================================

class ActionFFNEncoder(nn.Module):
    """MLP encoder: (B, T, action_dim) -> (B, T, embed_dim).

    Output tokens replace the text context in DiT cross-attention.

    With `drop_action=True` the MLP is bypassed and a learnable `null_token` is
    broadcast over the sequence instead. That is the "this task has no action"
    state, used by the video-conditioned forcing tasks (see TASK_SPECS in
    diffsynth/diffusion/flow_match_acwm_forcing.py).

    The null token exists because a zero action vector would NOT mean "no
    action": actions are normalized by (a - mean) / std, so zeros decode to the
    dataset-mean action -- in particular gripper_abs = -1.717, not 0. That is a
    specific, confident condition. The DiT also requires a context of shape
    (B, T_action, dim) for its framewise action cross-attention, so something
    must be supplied; a learned symbol is the honest choice.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        num_layers: int = 2,
        max_timesteps: int = 16,
    ):
        super().__init__()

        layers = [nn.Linear(action_dim, embed_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

        pe = self._sinusoidal_pe(max_timesteps, embed_dim)
        self.temporal_pe = nn.Parameter(pe)  # (max_timesteps, embed_dim)

        # "This task has no action" token, broadcast over the sequence.
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        actions: torch.Tensor,
        drop_action: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim)
            drop_action:
                If True, emit the learnable null token instead of encoding
                `actions`. `actions` is still required, for batch size, length,
                dtype and device.

        Returns:
            (B, T, embed_dim)
        """
        B, T = actions.shape[0], actions.shape[1]

        # Both branches are always evaluated, and both always contribute to the
        # graph (one with a zero coefficient). `mlp` here is a single
        # Linear(action_dim, embed_dim), so the discarded branch is free, and
        # this keeps DDP from flagging `mlp` as unused on no-action tasks, or
        # `null_token` as unused on action tasks -- which would otherwise
        # require find_unused_parameters=True and a full graph traversal every
        # step.
        mlp_out = self.mlp(actions)

        # The null token keeps the same temporal PE and norm as the conditioned
        # path, so it stays in the distribution the DiT expects. The framewise
        # action cross-attention groups these tokens 4-per-latent-frame, so
        # preserving the PE preserves that indexing.
        null_out = self.null_token.expand(B, T, -1).to(
            dtype=mlp_out.dtype,
            device=mlp_out.device,
        )

        if drop_action:
            x = null_out + 0.0 * mlp_out
        else:
            x = mlp_out + 0.0 * null_out

        x = x + self.temporal_pe[:T].to(
            dtype=x.dtype,
            device=x.device,
        )

        return self.norm(x)


# ============================================================================
# Main
# ============================================================================

def main():
    # Hyperparameters
    batch_size = 2
    timesteps = 16
    action_dim = 7
    embed_dim = 512

    # Random input
    actions = torch.randn(batch_size, timesteps, action_dim)

    # Model
    model = ActionFFNEncoder(
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=2,
        max_timesteps=32,
    )

    # Forward
    outputs = model(actions)

    print(f"Input shape : {actions.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output dtype: {outputs.dtype}")


if __name__ == "__main__":
    main()