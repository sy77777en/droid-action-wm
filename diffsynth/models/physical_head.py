"""
physical_head.py

PhysicalHead: maps a DiT block-output hidden (A1 tap) to a physical coordinate
latent (contact OR force), predicting the CLEAN, SCALED physical latent.

One head per modality (train contact first, then force) — instantiate with the
modality you're training; out_ch is always 48 (matches your Physical VAE z_dim).

Grid facts (Wan2.2-TI2V-5B, 17 frames @ 480x640, patch_size (1,2,2)):
  - Physical VAE latent grid : 48 x 5 x 30 x 40   (target space)
  - DiT token grid           : f=5, h=15, w=20  -> L = 1500
  - A1 hidden                : (B, 1500, 3072)
The head reshapes (B,1500,3072) -> (B,3072,5,15,20), mixes locally, then
UPSAMPLES 2x2 spatially -> (B,*,5,30,40), and projects to 48 channels.

Target convention (decided): the head regresses the SCALED latent
  z_scaled = (mu - mean) * (1/std)
where mu = PhysicalVAE.encode_mu_logvar(GT)[0] and (mean,std) are the physical
VAE's finetuned per-channel latent stats. At inference: head_out -> unscale
(z = head_out/scale[1] + scale[0]) -> PhysicalVAE.decode_latent(z).
The head itself is scale-agnostic: it just learns whatever target you feed it.
Keeping the target scaled (~N(0,1) per channel) keeps the MSE well-conditioned.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal embedding. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=t.device))
        * torch.arange(half, device=t.device).float() / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PhysicalHead(nn.Module):
    """DiT hidden (B, L, dit_dim) -> physical latent (B, 48, T, H, W).

    L must equal t_lat * h_tok * w_tok (the DiT TOKEN grid, coarser than the
    physical latent by `spatial_up` in H and W). The head upsamples H,W by
    `spatial_up` to reach the physical latent grid.
    """

    def __init__(
        self,
        dit_dim: int = 3072,
        out_ch: int = 48,
        t_lat: int = 5,
        h_lat: int = 30,
        w_lat: int = 40,
        patch_spatial: int = 2,
        hidden: int = 1024,
        t_embed_dim: int = 256,
    ):
        """
        Grid is derived from the PHYSICAL LATENT grid (t_lat, h_lat, w_lat),
        which equals the video latent grid = (num_frames-1)//4+1, H/16, W/16.
        The DiT token grid is coarser by `patch_spatial` (TI2V-5B patch (1,2,2)),
        so token grid = (t_lat, h_lat//patch_spatial, w_lat//patch_spatial) and
        the head upsamples H,W by `patch_spatial` to reach the latent grid.

        e.g. 640x480, 17f -> latent (5,30,40), patch 2 -> token (5,15,20),
        upsample 2x -> (5,30,40). Change resolution -> just pass new h_lat/w_lat;
        nothing is hard-coded.
        """
        super().__init__()
        assert h_lat % patch_spatial == 0 and w_lat % patch_spatial == 0, (
            f"latent grid ({h_lat},{w_lat}) not divisible by patch_spatial "
            f"{patch_spatial}")
        self.out_ch = out_ch
        self.t_lat = t_lat
        self.h_lat, self.w_lat = h_lat, w_lat
        self.h_tok = h_lat // patch_spatial
        self.w_tok = w_lat // patch_spatial
        self.spatial_up = patch_spatial

        # 1) token-wise projection 3072 -> hidden
        self.in_proj = nn.Linear(dit_dim, hidden)
        self.act = nn.GELU()

        # 2) FiLM timestep modulation (head sees the noise level)
        self.t_embed_dim = t_embed_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),     # -> (scale, shift)
        )

        # 3) local spatial-temporal context on the coarse (t,h_tok,w_tok) grid
        self.spatial_conv = nn.Conv3d(hidden, hidden, kernel_size=3,
                                      padding=1, groups=hidden)

        # 4) 2x2 spatial upsample on the H,W plane (per latent-frame)
        self.upsample = nn.ConvTranspose2d(
            hidden, hidden, kernel_size=self.spatial_up, stride=self.spatial_up)

        # 5) project to 48 channels, ZERO-INIT (head starts as a no-op)
        self.out_proj = nn.Conv3d(hidden, out_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden.shape
        assert L == self.t_lat * self.h_tok * self.w_tok, (
            f"token count {L} != {self.t_lat}*{self.h_tok}*{self.w_tok}="
            f"{self.t_lat*self.h_tok*self.w_tok}; A1 grid mismatch "
            f"(check patch_size / num_frames / resolution)")

        # token-wise proj
        x = self.act(self.in_proj(hidden))                  # (B, L, hidden)

        # FiLM timestep
        te = timestep_embedding(timestep, self.t_embed_dim).to(x.dtype)
        scale, shift = self.t_mlp(te).chunk(2, dim=-1)      # (B, hidden) each
        x = x * (1 + scale[:, None]) + shift[:, None]

        # (B, L, hidden) -> (B, hidden, t, h_tok, w_tok)
        x = x.transpose(1, 2).reshape(
            B, -1, self.t_lat, self.h_tok, self.w_tok)
        x = self.act(self.spatial_conv(x))

        # 2x2 spatial upsample, per latent-frame
        Bn, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(Bn * T, C, H, W)     # (B*T, C, h, w)
        x = self.upsample(x)                                      # (B*T, C, 2h, 2w)
        x = x.reshape(Bn, T, C, H * self.spatial_up, W * self.spatial_up)
        x = x.permute(0, 2, 1, 3, 4).contiguous()                # (B, C, T, 2h, 2w)

        out = self.out_proj(x)                              # (B, 48, T, 30, 40)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    head = PhysicalHead(dit_dim=3072, out_ch=48,
                        t_lat=5, h_lat=30, w_lat=40, patch_spatial=2, hidden=256)
    hidden = torch.randn(1, 5 * 15 * 20, 3072)
    t = torch.tensor([500.0])
    out = head(hidden, t)
    print("hidden:", tuple(hidden.shape), "+ t -> out:", tuple(out.shape))
    assert out.shape == (1, 48, 5, 30, 40), out.shape

    # zero-init check: out must be exactly 0 before any training
    assert out.abs().max().item() == 0.0, "out_proj must be zero-init (no-op start)"
    print("zero-init OK: head output is all zeros at init (smooth onset)")

    # gradient sanity. NOTE: with out_proj zero-init, the FIRST backward gives
    # zero grad to in_proj (d_loss/d_in_proj ∝ out_proj.weight = 0), which is
    # expected. out_proj itself DOES receive grad (it connects directly to the
    # loss); once it updates to nonzero, later steps flow grad to in_proj.
    out2 = head(hidden, t)
    loss = (out2 - torch.randn_like(out2)).pow(2).mean()
    loss.backward()
    g_out = head.out_proj.weight.grad
    assert g_out is not None and g_out.abs().sum() > 0, "no grad to out_proj"
    print("grad OK (step1): out_proj receives gradient (in_proj still 0 by zero-init)")

    # simulate one optimizer step on out_proj, then check grad reaches in_proj
    with torch.no_grad():
        head.out_proj.weight += 0.01 * torch.randn_like(head.out_proj.weight)
    head.zero_grad()
    out3 = head(hidden, t)
    loss3 = (out3 - torch.randn_like(out3)).pow(2).mean()
    loss3.backward()
    g_in = head.in_proj.weight.grad
    assert g_in is not None and g_in.abs().sum() > 0, "no grad to in_proj after out_proj nonzero"
    print("grad OK (step2): once out_proj is nonzero, grad flows to in_proj")
    print("ALL CHECKS PASSED")