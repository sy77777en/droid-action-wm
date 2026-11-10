"""
physical_vae.py

Contact / Force VAE for InterSim, built on top of Wan2.2 VAE
(WanVideoVAE38).

Design
------
- Reuse the full Wan VAE backbone with z_dim=48 and the same 4x16x16
  compression ratio. The latent shape stays aligned with the video VAE:
  (B, 48, T', H/16, W/16).

- Only replace the first and last convolution layers:
    encoder.conv1     : IN_CH * patch^2 -> dims0
    decoder.head[-1]  : out_dim -> IN_CH * patch^2

- All middle blocks remain unchanged.

- New convolution weights are initialized from pretrained RGB weights by
  averaging over the RGB channel groups.

- contact: IN_CH = 2
    left-finger contact map + right-finger contact map

- force: IN_CH = 6
    left-finger fx, fy, fz + right-finger fx, fy, fz

- During reconstruction training, use encode_mu_logvar / decode_latent
  without applying self.scale. The scale is only needed when exporting
  latents for DiT training or inference.

Dependency
----------
Adjust this import according to your actual Wan VAE file path:

    from wan_vae import WanVideoVAE38, CausalConv3d, patchify, unpatchify
"""


import torch
import torch.nn as nn

from .wan_video_vae import WanVideoVAE38, CausalConv3d, patchify, unpatchify


class PhysicalVAE(WanVideoVAE38):
    """Base class shared by ContactVAE and ForceVAE."""

    IN_CH = None
    PATCH = 2

    def __init__(self, z_dim=48, dim=160):
        assert self.IN_CH is not None, "Subclasses must define IN_CH"
        super().__init__(z_dim=z_dim, dim=dim)

    # ------------------------------------------------------------------ #
    # Construction / pretrained weight initialization
    # ------------------------------------------------------------------ #

    @classmethod
    def from_wan(
        cls, 
        wan_vae, 
        z_dim=48, 
        dim=160, 
        trainable=True
    ):
        vae = cls(z_dim=z_dim, dim=dim)
        vae.model.load_state_dict(wan_vae.model.state_dict(), strict=True)
        vae._adapt_channels()
        if trainable:
            vae.model.requires_grad_(True)
            vae.model.train()
        return vae
    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path,
        z_dim=48,
        dim=160,
        device="cpu",
        trainable=True,
        map_location="cpu",
    ):
        """
        Load the original RGB Wan VAE first, then adapt the input/output
        channel dimensions for the target physical modality.

        Important order:
            1. Build the original RGB-compatible VAE.
            2. Load pretrained Wan VAE weights.
            3. Replace encoder.conv1 and decoder.head[-1].
            4. Initialize the new layers from RGB weights.

        If your current Wan VAE checkpoint loading logic is different,
        replace the two checkpoint-loading lines below. The key idea is
        still the same: load RGB weights first, then call _adapt_channels().
        """
        vae = cls(z_dim=z_dim, dim=dim)

        sd = torch.load(ckpt_path, map_location=map_location)
        sd = WanVideoVAE38.state_dict_converter().from_civitai(sd)
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[PhysicalVAE] missing={len(missing)} unexpected={len(unexpected)}")
        vae._adapt_channels()
        vae.to(device)
        if trainable:
            vae.model.requires_grad_(True)
            vae.model.train()
        return vae

    def _adapt_channels(self):
        """encoder.conv1 changes in-dim; decoder.head[-1] changes out-dim.
        The two reshape paths differ -- do not copy-paste one onto the other."""
        model = self.model
        old_in = 3
        p2 = self.PATCH * self.PATCH        # = 4
        in_ch = self.IN_CH
 
        # ---- encoder.conv1 : (3*4=12 -> dims0)  =>  (in_ch*4 -> dims0) ----
        old = model.encoder.conv1                  # CausalConv3d(12, dims0, 3, pad=1)
        dims0 = old.out_channels
        k = old.kernel_size                        # (3, 3, 3)
        new = CausalConv3d(in_ch * p2, dims0, k, padding=1)
        with torch.no_grad():
            w = old.weight.data                    # (dims0, 12, kt, kh, kw)
            w = w.view(dims0, old_in, p2, *k)      # (dims0, 3, 4, ...)
            w = w.mean(dim=1, keepdim=True)        # average over RGB -> (dims0, 1, 4, ...)
            w = w.repeat(1, in_ch, 1, 1, 1, 1)     # (dims0, in_ch, 4, ...)
            new.weight.data.copy_(w.reshape(dims0, in_ch * p2, *k))
            new.bias.data.copy_(old.bias.data)     # bias is on out-dim, unchanged
        model.encoder.conv1 = new
 
        # ---- decoder.head[-1] : (out_dim -> 12)  =>  (out_dim -> in_ch*4) ----
        oldh = model.decoder.head[-1]              # CausalConv3d(out_dim, 12, 3, pad=1)
        out_dim = oldh.in_channels
        kh = oldh.kernel_size
        newh = CausalConv3d(out_dim, in_ch * p2, kh, padding=1)
        with torch.no_grad():
            w = oldh.weight.data                   # (12, out_dim, ...)
            w = w.view(old_in, p2, out_dim, *kh)   # (3, 4, out_dim, ...)
            w = w.mean(dim=0, keepdim=True)        # average over RGB -> (1, 4, out_dim, ...)
            w = w.repeat(in_ch, 1, 1, 1, 1, 1)     # (in_ch, 4, out_dim, ...)
            newh.weight.data.copy_(w.reshape(in_ch * p2, out_dim, *kh))
            b = oldh.bias.data.view(old_in, p2).mean(0, keepdim=True)  # (1, 4)
            newh.bias.data.copy_(b.repeat(in_ch, 1).reshape(-1))
        model.decoder.head[-1] = newh

    # ------------------------------------------------------------------ #
    # Training forward path
    # ------------------------------------------------------------------ #
    def encode_mu_logvar(self, x):
        """
        Encode input physical maps into latent distribution parameters.

        Args:
            x: Tensor with shape (B, IN_CH, T, H, W).
               The input should already be normalized by the dataloader.

        Returns:
            mu:      Tensor with shape (B, 48, T', H/16, W/16).
            log_var: Tensor with shape (B, 48, T', H/16, W/16).

        Note:
            This function does not apply self.scale.
        """
        m = self.model
        m.clear_cache()
        x = patchify(x, patch_size=self.PATCH)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None
        for i in range(iter_):
            m._enc_conv_idx = [0]
            seg = x[:, :, :1] if i == 0 else x[:, :, 1+4*(i-1): 1+4*i]
            o, m._enc_feat_map, m._enc_conv_idx = m.encoder(
                seg, feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx)
            out = o if out is None else torch.cat([out, o], dim=2)
        mu, log_var = m.conv1(out).chunk(2, dim=1)
        m.clear_cache()
        return mu, log_var

    # =====================================================================
    # ADD THIS METHOD to the PhysicalVAE class in
    #   diffsynth/models/physical_vae.py
    # (put it next to encode_mu_logvar / decode_latent)
    # =====================================================================
 
    @torch.no_grad()
    def encode_for_dit(self, x, sample=False, apply_scale=True):
        """
        Encode a physical map clip into a DiT-ready latent.
 
        Args:
            x: (B, IN_CH, T, H, W) float tensor, ALREADY normalized by the
               dataloader with the exact same normalization used to train
               this VAE (per-channel scale, force_clip, contact_gain).
            sample: if True, sample from N(mu, sigma); if False, use mu
                    (deterministic, correct for training the DiT).
            apply_scale: if True, normalize the latent with self.scale so its
                    per-channel distribution matches the video latent scale.
                    self.scale is [mean(48,), 1/std(48,)]; latent = (z-mean)/std.
                    Default self.scale (mean=0, inv_std=1) is identity.
 
        Returns:
            latent: (B, 48, T', H/16, W/16)
        """
        mu, log_var = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, log_var) if sample else mu
 
        if apply_scale and getattr(self, "scale", None) is not None:
            mean, inv_std = self.scale[0], self.scale[1]
            if not torch.is_tensor(mean):
                mean = torch.as_tensor(mean)
            if not torch.is_tensor(inv_std):
                inv_std = torch.as_tensor(inv_std)
            mean = mean.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
            inv_std = inv_std.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
            z = (z - mean) * inv_std
 
        return z

    
    def decode_latent(self, z, unscale=False):
        """
        Decode a physical latent back to physical maps.

        Args:
            z: Tensor with shape (B, 48, T', H/16, W/16).
            unscale: set True when z came from the SCALED latent space --
                i.e. from vae.encode() / encode_for_dit(apply_scale=True), or
                from the DiT, which is trained on those latents. Applies the
                same inverse WanVideoVAE.model.decode() does:
                    z = z / scale[1] + scale[0]      # scale = [mean, 1/std]
                Leave False when z came from encode_mu_logvar(), which does
                not scale -- that is what PhysicalVAE.forward() uses.

        Returns:
            Reconstructed tensor with shape (B, IN_CH, T, H, W).

        Why this flag exists:
            vae.encode() -> model.encode() normalizes the latent
            (mu = (mu - mean) / std), and the symmetric model.decode()
            inverts it. decode_latent() bypasses model.decode() -- it has to,
            because model.decode() ends in .clamp_(-1, 1), which is right for
            RGB but wrong for contact_gain=5.0 / force_clip=8.0 physical maps.
            Bypassing it also dropped the unscale, so every inference path that
            fed a scaled latent here decoded it in the wrong space. Measured on
            a GT round-trip, that turned a sparse contact map (active
            fraction 0.019) into a diffuse one (0.509); with unscale=True the
            same round-trip returns 0.029, correlation 1.000, peak ratio 1.003.
        """
        if unscale:
            sc = getattr(self, "scale", None)
            if sc is None:
                raise ValueError(
                    "decode_latent(unscale=True) needs self.scale; load the "
                    "<modality>_latent_scale.pt next to the VAE checkpoint."
                )
            mean = torch.as_tensor(sc[0]).to(device=z.device, dtype=z.dtype)
            inv_std = torch.as_tensor(sc[1]).to(device=z.device, dtype=z.dtype)
            c = z.shape[1]
            z = z / inv_std.view(1, c, 1, 1, 1) + mean.view(1, c, 1, 1, 1)

        m = self.model
        m.clear_cache()
        x = m.conv2(z)
        out = None
        for i in range(z.shape[2]):
            m._conv_idx = [0]
            o, m._feat_map, m._conv_idx = m.decoder(
                x[:, :, i:i + 1], feat_cache=m._feat_map,
                feat_idx=m._conv_idx, first_chunk=(i == 0))
            out = o if out is None else torch.cat([out, o], dim=2)
        out = unpatchify(out, patch_size=self.PATCH)
        m.clear_cache()
        return out

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def forward(self, x, sample=True):
        mu, log_var = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, log_var) if sample else mu
        x_recon = self.decode_latent(z)
        return x_recon, mu, log_var

    # ------------------------------------------------------------------ #
    # Recompute latent scale after finetuning
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def recompute_latent_scale(self, loader, device="cuda", max_batches=None):
        """
        Recompute per-channel latent mean/std on the physical modality data.

        This should be done after finetuning. The resulting scale should be
        used when exporting physical latents for DiT training.

        Args:
            loader: Dataloader that returns either:
                - Tensor with shape (B, IN_CH, T, H, W), or
                - Dict containing {"data": tensor}.
            device: Device used for computation.
            max_batches: Optional limit on the number of batches.

        Returns:
            mean: Tensor with shape (48,).
            std: Tensor with shape (48,).
        """
        self.model.eval()
        sums = torch.zeros(self.z_dim, device=device)
        sqs = torch.zeros(self.z_dim, device=device)
        cnt = 0
        for bi, batch in enumerate(loader):
            x = batch if torch.is_tensor(batch) else batch["data"]
            x = x.to(device)
            mu, _ = self.encode_mu_logvar(x)                  # (B,48,T',h,w)
            v = mu.permute(1, 0, 2, 3, 4).reshape(self.z_dim, -1)
            sums += v.sum(1); sqs += (v * v).sum(1); cnt += v.shape[1]
            if max_batches and bi + 1 >= max_batches:
                break
        mean = sums / cnt
        std = (sqs / cnt - mean ** 2).clamp_min(1e-8).sqrt()
        self.mean = mean.detach().cpu()
        self.std = std.detach().cpu()
        self.scale = [self.mean, 1.0 / self.std]
        return self.mean, self.std


class ContactVAE(PhysicalVAE):
    """
    Contact map VAE.

    Input shape:
        (B, 2, T, H, W)

    Channels:
        ch0: left-finger contact map
        ch1: right-finger contact map
    """

    IN_CH = 2


class ForceVAE(PhysicalVAE):
    """
    Force field VAE.

    Input shape:
        (B, 6, T, H, W)

    Channels:
        ch0-ch2: left-finger fx, fy, fz
        ch3-ch5: right-finger fx, fy, fz
    """

    IN_CH = 6


# ---------------------------------------------------------------------- #
# Sanity check: verify latent and reconstruction shapes.
# This does not require pretrained weights.
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for Cls, C in [(ContactVAE, 2), (ForceVAE, 6)]:
        vae = Cls()
        vae._adapt_channels()
        vae = vae.to(device)
        vae.model.eval()                       # shape check only, no training

        # Shape check only -> no autograd. Keeping grads on the causal decode
        # loop retains the full graph + feat_cache activations and can blow up
        # GPU memory (this is what OOM'd an 80GB card). Real training is fine
        # with grads because it uses batch=1 / gradient checkpointing.
        with torch.no_grad():
            x = torch.randn(1, C, 17, 480, 640, device=device)   # 17 = 4k + 1
            mu, log_var = vae.encode_mu_logvar(x)
            z = vae.reparameterize(mu, log_var)
            rec = vae.decode_latent(z)

        print(
            f"{Cls.__name__}: "
            f"input {tuple(x.shape)} -> "
            f"latent {tuple(mu.shape)} -> "
            f"recon {tuple(rec.shape)}"
        )

        assert mu.shape[1] == 48, "Latent channel dimension must be 48"
        assert mu.shape[3] == 480 // 16, "Height must be compressed by 16x"
        assert mu.shape[4] == 640 // 16, "Width must be compressed by 16x"
        assert rec.shape == x.shape, "Reconstruction shape must match input"

        del vae, x, mu, log_var, z, rec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("OK: latent alignment and reconstruction shape checks passed")
        
