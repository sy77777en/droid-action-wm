#!/usr/bin/env bash
# Downloads (1) our checkpoint + action stats + example clips and (2) the Wan2.2-TI2V-5B
# DiT shards and VAE (the DiT weights are only used to build the architecture; every DiT
# tensor is then overwritten by our checkpoint). ~10 GB + ~23 GB.
set -euo pipefail
DEST="${1:-weights}"
mkdir -p "${DEST}"
hf download syCen/droid-action-wm --local-dir "${DEST}/droid-action-wm"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "${DEST}/Wan2.2-TI2V-5B" \
  --include "diffusion_pytorch_model-0000?-of-00003.safetensors" \
            "diffusion_pytorch_model.safetensors.index.json" "config.json" "Wan2.2_VAE.pth"
echo "done -> ${DEST}"
