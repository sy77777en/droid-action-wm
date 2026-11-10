# DROID Action-Conditioned World Model

An action-conditioned video world model for the [DROID](https://droid-dataset.github.io/) Franka setup.
Given **one exterior-camera image** and a **chunk of 16 end-effector actions**, it predicts the next
**16 frames at 5 Hz** (3.2 s) at 640×480. The intended use is **offline policy evaluation**: roll a
policy out inside the model instead of on the robot.

| | |
|---|---|
| Backbone | Wan2.2-TI2V-5B (5B DiT + Wan2.2 VAE), fully fine-tuned |
| Training data | all of DROID 1.0.1 (train split; ~2% of scenes and ~1% of episodes held out), exterior cameras, no text |
| Checkpoint | `droid_wm_fullft_s64730.safetensors` (bf16, 10 GB) on 🤗 [`syCen/droid-action-wm`](https://huggingface.co/syCen/droid-action-wm) |
| Inference | 1 GPU, 50 denoising steps, ~13 s per 16-frame chunk on an RTX PRO 6000 (tested); peak VRAM not yet profiled |

## 1. Install

```bash
git clone <this repo> && cd droid-action-wm
conda create -n droidwm python=3.10 -y && conda activate droidwm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt
```

The `diffsynth/` directory is a vendored, modified copy of
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache-2.0) that contains the
model code. Use it from the repo root; **don't** `pip install diffsynth`, because an installed upstream
copy would shadow it.

## 2. Download weights

```bash
huggingface-cli login            # the HF repo is private for now
bash scripts/download_weights.sh weights
```

This creates:

```
weights/
├── droid-action-wm/
│   ├── droid_wm_fullft_s64730.safetensors   # our model (all DiT + action-encoder weights)
│   ├── action_stats_full_interp.json        # action normalisation (must match the ckpt)
│   └── examples/clip{0,1,2}/                # 3 held-out-scene clips for a sanity check
└── Wan2.2-TI2V-5B/                          # DiT shards (architecture only) + Wan2.2_VAE.pth
```

## 3. Check that your install matches ours

```bash
python scripts/check_examples.py --weights weights
# clip0: PSNR 22.95 dB (paper eval: 22.95)
# clip1: PSNR 22.83 dB (paper eval: 22.83)
# clip2: PSNR 23.06 dB (paper eval: 23.06)
```

Each example uses the seed from our eval, so the PSNR should match the reference to within a few
hundredths of a dB. Small drift across GPU types and library versions is normal; a gap of several dB
means something is wrong (usually a wrong stats file or image preprocessing).

## 4. Predict

**Command line**

```bash
python scripts/predict.py --weights weights \
    --obs weights/droid-action-wm/examples/clip0/obs.png \
    --actions weights/droid-action-wm/examples/clip0/actions_raw.npy \
    --out out/clip0           # frame_01..16.png + pred.mp4
```

You can pass `--poses poses.npy --gripper gripper.npy` (absolute EEF poses for frames 0..16) instead
of `--actions`. If the actions have shape `(16*K, 7)`, the script runs an autoregressive rollout.

**Python**

```python
from PIL import Image
from droid_wm import DroidWorldModel, actions_from_poses

wm = DroidWorldModel.from_pretrained(
    ckpt="weights/droid-action-wm/droid_wm_fullft_s64730.safetensors",
    wan_root="weights/Wan2.2-TI2V-5B",
    stats_json="weights/droid-action-wm/action_stats_full_interp.json")

frames = wm.predict(Image.open("obs.png"), actions)   # actions: (16, 7) raw rows -> 16 PIL frames
```

**From a raw DROID episode** (DROID 1.0.1 raw layout, `trajectory.h5` + `recordings/MP4/`):

```bash
python scripts/clip_from_droid.py --episode /path/to/episode --cam ext1 --start 60 --out clip/
python scripts/predict.py --obs clip/obs.png --actions clip/actions_raw.npy --out clip/pred
```

## 5. Input conventions (read this before plugging in a policy)

**Image**
- One DROID **exterior** camera, left image (`ext1` or `ext2`).
- The image is centre-cropped to 4:3 and resized to 640×480 (LANCZOS). `predict()` does this for you.
- The model does **not** use the wrist camera or a language instruction.

**Timing**
- DROID runs at 15 Hz; the model runs at 5 Hz.
- Model frame *j* is raw step *s + 3j*.

**Action row *j*** (*j* = 0..15) is 7 numbers in the **robot base frame**, un-normalised:

| dims | meaning |
|---|---|
| 0–2 | `xyz(j+1) − xyz(j)`, metres |
| 3–5 | `euler(j+1) − euler(j)`, wrapped to (−π, π], radians (DROID `cartesian_position` Euler convention) |
| 6 | `gripper(j+1)`, absolute: 0 = open, 1 = closed |

- `xyz`/`euler` come from `observation/robot_state/cartesian_position`.
- `gripper` comes from `observation/robot_state/gripper_position`.
- Row *j* takes the arm from frame *j* to frame *j+1*, so it produces predicted frame *j+1*.
- `actions_from_poses(cartesian[17,6], gripper[17])` builds the rows for you.
- Training interpolated poses to each frame's capture time (≈55 ms before the state read). Step-aligned poses are fine.

## 6. Policy evaluation recipe

Most DROID policies output joint velocities or joint/cartesian targets. The world model wants the
**resulting end-effector trajectory**. Run the policy's actions through forward kinematics or your
controller model to get EEF poses at 5 Hz, then:

```python
obs = first_frame                        # real exterior image at t=0
for chunk in range(K):
    poses, grip = policy_to_eef(policy, obs, ...)        # 17 poses: current + 16 future (5 Hz)
    frames = wm.predict(obs, actions_from_poses(poses, grip), seed=chunk)
    obs = frames[-1]                     # feed the imagined frame back to the policy
    log(frames)                          # then score success (VLM judge / human / heuristic)
```

Things to keep in mind:
- **One view only.** If your policy needs the wrist camera, the model can't supply it. Use exterior-only policies, or feed the wrist input from elsewhere.
- **Long horizons drift.** Every chunk conditions on a generated frame, so errors compound. Within one 16-frame chunk the model is accurate (table below). On our long-horizon test (an earlier checkpoint of this run, 64 frames = 4 chunks), PSNR fell from 27.1 dB at frame 1 to 19.4 at frame 16 and 16.9 at frame 64. That is still above a frozen-frame baseline, but Ctrl-World drifts less over 64 frames. Keep rollouts to a few chunks, or re-anchor on real frames when you can.
- **Sampling is stochastic.** Different seeds give different futures. For success-rate estimates, average over several seeds.

## 7. Accuracy

Accuracy is measured on 16 predicted frames (5 Hz), 640×480, 60 clips per set, 50 steps, seed = clip index. The
end-effector stays ≥100 px inside the crop for all 17 frames.

| eval set | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| train episodes | 22.33 | 0.850 | 0.092 |
| held-out episodes (seen scenes) | 21.76 | 0.841 | 0.093 |
| held-out scenes | 22.61 | 0.859 | 0.082 |

SSIM is standard SSIM (7×7 windows, RGB).

## Files

```
droid_wm/model.py         DroidWorldModel: from_pretrained / predict / rollout
droid_wm/actions.py       action rows, normalisation, image preprocessing
droid_wm/_sampler.py      sampler + model loading (vendored from our research code)
scripts/                  download_weights.sh, predict.py, check_examples.py, clip_from_droid.py
diffsynth/                modified DiffSynth-Studio (Wan2.2 + action-conditioned MoT DiT)
```

## License

Code: Apache-2.0 (inherits DiffSynth-Studio). The weights are derived from Wan2.2-TI2V-5B (Apache-2.0)
and fine-tuned on DROID (CC-BY-4.0).
