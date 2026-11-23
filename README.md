<h1 align="center">Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth</h1>

<p align="center">
  <a href="<PROJECT_PAGE_URL>"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/ShimaRinC/Distill-to-Loop"><img src="https://img.shields.io/badge/Model-Weights-yellow" alt="Model"></a>
  <a href="https://arxiv.org/abs/<ARXIV_ID>"><img src="https://img.shields.io/badge/arXiv-<ARXIV_ID>-b31b1b" alt="arXiv"></a>
</p>

<!-- Teaser figure: to be added. -->

## Abstract

Looped transformers reuse one block of shared weights along the depth of the network. Recent looped diffusion transformers are trained from scratch, which repeats the full pretraining cost although strong pretrained non-looped diffusion transformers exist. We propose Distill to Loop (D2L), which distills a pretrained non-looped diffusion transformer into a looped student whose blocks are initialized from the teacher's. We cast distillation as a fixed-point iteration toward the teacher, in which the output after every loop is trained toward the teacher, and each loop is one step of an ordinary differential equation in depth. As a result, one model runs any number of loops at test time without retraining, and the higher-order ODE solvers of diffusion model sampling can be applied along the depth dimension for better results. Meanwhile, because a looped model produces an output after every loop, we also construct loop guidance between the outputs of different loop counts, which improves the generated samples at little extra compute. On SD3-medium, a looped student with about 37% of the teacher's parameters matches the teacher on average over GenEval, ImageReward, PickScore and HPSv2.1 with two to four loops, and with two loops it uses about 58% of the teacher's compute. And on a four-step SD3-medium, loop guidance raises the average of the four metrics from 88.2 to 99.2.

## What this repository contains

This repository covers the SD3-medium distillation and adjustable-loop inference: data preparation, training, export of the student, image generation, and a Python API. Loop guidance, the higher-order depth solvers and the other models of the paper are not included. The repository contains no evaluation code.

The teacher is SD3-medium (24 MMDiT blocks, 2.03B parameters). The student has 2 input blocks, 5 looped blocks that share weights, and 2 output blocks (756,415,672 trainable parameters, 0.756B). It is trained with K = 4 loops, and K is chosen at sampling time.

Results from the paper with SD3-medium as the teacher (28 sampling steps, CFG 7, 512 px). Avg. is the mean of the four text-to-image metrics, in percent relative to the teacher.

| Model | TFLOPs | Rounds K | GenEval overall | ImageReward | PickScore | HPSv2.1 | Avg. (%) |
|---|---|---|---|---|---|---|---|
| SD3-medium (teacher) | 50.5 | - | 0.648 | 0.908 | 22.214 | 0.278 | 100.0 |
| D2L | 18.9 | 1 | 0.589 | 0.663 | 21.790 | 0.259 | 88.8 |
| D2L | 29.4 | 2 | 0.661 | 0.924 | 22.245 | 0.275 | 100.7 |
| D2L | 39.9 | 3 | 0.654 | 0.933 | 22.307 | 0.276 | 100.9 |
| D2L | 50.5 | 4 | 0.659 | 0.919 | 22.312 | 0.276 | 100.7 |
| D2L | 92.6 | 8 | 0.648 | 0.889 | 22.278 | 0.275 | 99.3 |
| D2L | 177.0 | 16 | 0.633 | 0.853 | 22.229 | 0.273 | 97.6 |

The table is the plain distilled student, which the default config reproduces. The Hugging Face repository holds `d2l-sd3-medium` (the distilled student combined with a real-data continuation, the "Real data" rows of the paper) and `d2l-sd3-medium-turbo` (distilled from the CDM four-step teacher); see its model card.

## Getting Started

### 1. Install

The commands in this README use these shell variables:

```bash
export D2L_VENV=/path/to/venv                                # virtual environment directory
export WORK=/path/to/work                                    # scratch directory for data, checkpoints and outputs
export COCO_IMAGES=/path/to/coco/train2017                   # COCO 2017 train2017/ image directory
export COCO_ANNOTATIONS=/path/to/coco/annotations/captions_train2017.json   # COCO 2017 caption annotations
export TRAINED_CHECKPOINT=/path/to/step_00100000.pt          # a training checkpoint to export
```

```bash
python -m venv "$D2L_VENV"
source "$D2L_VENV/bin/activate"
pip install -r requirements.txt
```

Alternatively, `pip install -e .` installs the package `distill-to-loop` 0.1.0 with the same pinned dependencies (extras `test` and `wandb`). Python 3.10 or newer is required, and the install was checked with Python 3.11. All commands below run from the repository root.

### 2. Prepare weights and data

The code never downloads the teacher, and the loaders read the local Hugging Face cache only. Accept the license of SD3-medium on the Hugging Face Hub, log in, and download the teacher once (`HF_HUB_CACHE` selects the cache, or pass `--hf-cache`):

```bash
hf auth login
hf download stabilityai/stable-diffusion-3-medium-diffusers
```

With huggingface_hub older than 1.0, use `huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers`. If the teacher is missing, the scripts stop with a `FileNotFoundError` that lists these steps. The training launcher sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` as defaults; a value set in the environment overrides them. Student weights: https://huggingface.co/ShimaRinC/Distill-to-Loop.

Training uses COCO 2017 train at 512 px. Download `train2017.zip` and `annotations_trainval2017.zip` from https://cocodataset.org/#download; the commands use the `train2017/` image directory and `annotations/captions_train2017.json`. `scripts/prepare_coco.py` encodes one caption per image with the SD3 VAE and text encoders and writes the shards that the trainer reads. The full split (118,287 images, 119 shards) takes about 182 GB on disk. Example with 2 GPUs:

```bash
torchrun --nproc_per_node=2 scripts/prepare_coco.py --images-root "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
    --out "$WORK/coco2017_512/train"
```

For a quick check on the CPU, the first 16 images can be prepared with:

```bash
python scripts/prepare_coco.py --images-root "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
    --out "$WORK/coco2017_512/train" --limit 16 --shard-size 8 --batch 4 --text-batch 8 --workers 2 --device cpu
```

### 3. Train

The recipe is in `configs/sd3m_d2l.yaml`. The original run used 2 x NVIDIA B200 with 72 samples per GPU (144 per step: 96 data states and 48 rollout states) for 100k steps:

| Setting | Value |
|---|---|
| Data | COCO 2017 train, 512 px, condition dropout 0.1 |
| Student | 2 input + 5 looped + 2 output blocks, K = 4 |
| Optimizer | AdamW, betas (0.9, 0.999), no weight decay, gradient clip 1.0 |
| Learning rate | constant 1.732e-4, 500 warm-up steps |
| Steps | 100k, EMA 0.9995 |
| Precision | bf16 autocast, fp32 master weights |
| Online rollout | fraction 1/3, 8 Euler steps, teacher CFG 4.5, w_c = 2 |
| Input-block loss weight | lambda_0 = 0.1 |
| Attention | QK normalization in the looped and output blocks; soft exclusive self-attention in the looped blocks |

```bash
GPUS=0,1 CONFIG=configs/sd3m_d2l.yaml OVERRIDE="data.train_dir=$WORK/coco2017_512/train" bash scripts/launch_train.sh
```

`GPUS` lists the GPU indices. `OVERRIDE` takes space-separated `key.path=value` overrides of the YAML config. The per-GPU batch size is set with `train.per_gpu_batch_size` in `OVERRIDE`. The layouts that keep the 144 rows per step of the recipe are 1 x 144, 2 x 72, 3 x 48 and 6 x 24 (GPUs x per-GPU batch), for example `GPUS=0,1,2 ... train.per_gpu_batch_size=48`. Any other total changes the global batch size, and the trainer prints a `[train] WARNING: ... rows per step ...` at start-up. The rollout rows per GPU, `round(per_gpu_batch_size / 3)`, must be a positive multiple of the 8 rollout steps (for example 24, 48 or 72 rows per GPU); other values are refused. There is no gradient accumulation.

The original run used 72 rows per GPU with round checkpointing (`train.loop_grad_checkpoint: true`) and peaked at 137.63 GiB per GPU, which is close to the capacity of 141 GB cards. Smaller cards need another layout or the flag `train.block_grad_checkpoint=true` (default `false`), which adds per-block activation checkpointing to reduce memory at the cost of more recomputation. On one H200 at 24 rows per GPU, the peak memory was 22.45 GiB with the flag and 56.50 GiB without it, at about 4.3 s and 3.5 s per step (the GPU was shared, so the timings are upper bounds). The flag gives bitwise the same results on the CPU (tested). On CUDA, training is not bit-reproducible from run to run, and the differences with the flag are of the same size as those between two runs without it. The flag may be changed when resuming.

Checkpoints are written to `<output_dir>/<name>/step_XXXXXXXX.pt`. `output_dir` defaults to `./checkpoints`; point it to a disk with room, because each checkpoint takes about 12 GB and the last `train.keep_last` (default 10) are kept.

To check the training pipeline on the CPU without the real teacher, `tools/make_tiny_teacher.py` creates a small random teacher with the input shapes of SD3-medium:

```bash
python tools/make_tiny_teacher.py --out "$WORK/tiny_teacher"
GPUS=cpu CONFIG=configs/sd3m_d2l.yaml \
OVERRIDE="name=smoke output_dir=$WORK/checkpoints data.train_dir=$WORK/coco2017_512/train data.num_workers=0 model.teacher_transformer_path=$WORK/tiny_teacher train.per_gpu_batch_size=24 train.max_steps=2 train.checkpoint_every=1 train.log_every=1" \
bash scripts/launch_train.sh
```

### 4. Export and generate

Export the EMA weights of a checkpoint as a student directory (`config.json` and `diffusion_pytorch_model.safetensors`):

```bash
python scripts/export_student.py --checkpoint "$TRAINED_CHECKPOINT" --out "$WORK/exported/sd3m_d2l_100k"
```

`--metadata KEY=VALUE` (repeatable) adds entries to the metadata of the exported weight file, for example how the weights were averaged after training. The standard entries (`teacher_model_id`, `training_step`, `ema_decay`, `weights`) cannot be replaced.

```bash
python scripts/export_student.py --checkpoint "$TRAINED_CHECKPOINT" --out "$WORK/exported/sd3m_d2l_100k" \
    --metadata "sampling=K 4, 28 Euler steps on the scheduler's grid, CFG 7.0, 512 px"
```

Generate images from a text file with one prompt per line. The loop count `--K` is chosen at sampling time. The defaults of `scripts/generate.py` are the released sampling setting `--K 4 --steps 28 --cfg 7.0 --size 512`. `--size` must be a multiple of 16; other sizes are rejected at argument parsing (exit status 2, `generate.py: error: --size must be a positive multiple of 16 (VAE factor 8 x patch size 2), got 520` for `--size 520`). The commands below use a smaller setting (256 px, 8 steps) and the CPU:

```bash
printf 'a red bicycle leaning against a brick wall\na bowl of fruit on a wooden table\n' > "$WORK/prompts.txt"
python scripts/generate.py --student "$WORK/exported/sd3m_d2l_100k" --prompts "$WORK/prompts.txt" --out "$WORK/samples/k4" \
    --K 4 --steps 8 --cfg 7.0 --size 256 --seed 0 --micro-batch 2 --device cpu
python scripts/generate.py --student "$WORK/exported/sd3m_d2l_100k" --prompts "$WORK/prompts.txt" --out "$WORK/samples/k2" \
    --K 2 --steps 8 --cfg 7.0 --size 256 --seed 0 --micro-batch 2 --device cpu
```

`--sigmas` replaces the `--steps`-point grid of the scheduler by custom sampling sigmas, given before the scheduler's shift; the step count then equals the number of sigmas, and a conflicting `--steps` is rejected. For example, the four-step student of the Hugging Face repository is sampled with:

```bash
--K 4 --sigmas 1.0 0.75 0.5 0.25 --cfg 1.0
```

Each run writes the images, `prompts.txt` and `generation.json` (the sampling settings and the metadata of the export) to `--out`. On a GPU, omit `--device` (it defaults to CUDA when available).

### 5. Inference in Python

```python
import os
import torch
from distill_to_loop import encode_prompts, generate_images, load_pipeline, load_student, load_vae_and_scheduler_factory

work = os.environ["WORK"]
student_dir = f"{work}/exported/sd3m_d2l_100k"
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
device = "cpu"  # "cuda" on a GPU

# The release protocol: prompts encoded once, per-image seeded noise, loop count K chosen at sampling time.
student = load_student(student_dir, device=device)
vae, scheduler_factory = load_vae_and_scheduler_factory(model_id, device)
encodings = encode_prompts(["a lighthouse on a cliff at sunset"], model_id, device)
for k in (1, 4):
    (_, _, image), = generate_images(student, vae, scheduler_factory, encodings, loop_rounds=k, steps=8, cfg_scale=7.0,
                                     resolution=256, seed=0, micro_batch_size=1, device=device)
    image.save(f"{work}/api_k{k}.png")

# A diffusers pipeline with the student as its transformer (T5 length 77 and, on CUDA, bf16 autocast of the transformer by default).
pipe = load_pipeline(student_dir, device=device)
pipe.transformer.loop_rounds = 4
image = pipe("a lighthouse on a cliff at sunset", num_inference_steps=8, guidance_scale=7.0, height=256, width=256,
             generator=torch.Generator(device).manual_seed(0)).images[0]
image.save(f"{work}/api_pipeline_k4.png")
print("parameters:", sum(p.numel() for p in pipe.transformer.parameters()))
```

`load_pipeline` returns a `LoopedSD3Pipeline` (a diffusers `StableDiffusion3Pipeline`). Its call uses `max_sequence_length=77`, the T5 length used in training (the diffusers default is 256; pass another value to override it), and on CUDA it runs every transformer forward under bf16 autocast while the student keeps fp32 weights, as in training. The text encoders and the VAE run without autocast, in bf16. The pipeline draws its noise from `generator`, so its images differ from those of `generate_images` and `scripts/generate.py`, whose noise is seeded per image. On CUDA, those two also decode the VAE under autocast.

## Resuming a run

Pass `RESUME_FROM` to the launcher with the same `CONFIG` and `OVERRIDE` as the original run:

```bash
GPUS=cpu CONFIG=configs/sd3m_d2l.yaml RESUME_FROM="$WORK/checkpoints/smoke/step_00000002.pt" \
OVERRIDE="name=smoke output_dir=$WORK/checkpoints data.train_dir=$WORK/coco2017_512/train data.num_workers=0 model.teacher_transformer_path=$WORK/tiny_teacher train.per_gpu_batch_size=24 train.max_steps=3 train.checkpoint_every=1 train.log_every=1" \
bash scripts/launch_train.sh
```

Starting a run again without `RESUME_FROM` in a directory that already holds checkpoints is refused, and so is resuming with a changed recipe field. Fields that may change on resume include the run length, the reporting cadence, the two checkpointing flags, the logging, and the locations (`name` and `output_dir`).

Resuming a checkpoint into a run directory that already holds later checkpoints is also refused, because the resumed run would overwrite or prune them. For example, with `smoke/` holding steps 1 to 3, resuming `step_00000001.pt` with `name=smoke` stops with `RuntimeError: cannot resume ... the directory already holds later checkpoints (step_00000002.pt, step_00000003.pt) ...`.

To resume an older checkpoint into a new, empty run directory, add `name=<new name>` to `OVERRIDE` (or use another `output_dir`):

```bash
GPUS=cpu CONFIG=configs/sd3m_d2l.yaml RESUME_FROM="$WORK/checkpoints/smoke/step_00000001.pt" \
OVERRIDE="name=smoke_from_step_1 output_dir=$WORK/checkpoints data.train_dir=$WORK/coco2017_512/train data.num_workers=0 model.teacher_transformer_path=$WORK/tiny_teacher train.per_gpu_batch_size=24 train.max_steps=3 train.checkpoint_every=1 train.log_every=1" \
bash scripts/launch_train.sh
```

Resuming into a directory other than the checkpoint's own is refused when that directory already holds checkpoints or a `wandb_run_id` file. For example, `smoke_from_step_1` now holds steps 2 and 3, so resuming `step_00000003.pt` of `smoke/` with `name=smoke_from_step_1` stops with:

```text
RuntimeError: cannot resume $WORK/checkpoints/smoke/step_00000003.pt into $WORK/checkpoints/smoke_from_step_1: the directory holds checkpoints of another run (step_00000002.pt, step_00000003.pt); pick a new name (name=...) or another output_dir=... for the resumed run (it would prune those checkpoints and log into that run's W&B run)
```

If a resume under a new name with W&B enabled stops before its first checkpoint, it leaves a `wandb_run_id` file in that directory and the same command is then refused; use another name or remove that directory.

A resumed run is exact only with `data.num_workers=0` (checked on the CPU). With data loader workers (the default is 8) the sample order is the same, but after a resume within an epoch the latents drawn from the stored VAE posteriors differ from those of the uninterrupted run.

## Tests

```bash
python -m pytest -q
```

The suite has 49 tests and hides the GPUs itself.

## Related links

- Teacher model: [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- Project page: `<PROJECT_PAGE_URL>`
- Student weights: `https://huggingface.co/ShimaRinC/Distill-to-Loop`

## License

The code is released under the Apache-2.0 license (see `LICENSE`). Third-party components are listed in `NOTICE` and keep their own licenses; for example, `distill_to_loop/models/looped_sd3.py` follows diffusers' `SD3Transformer2DModel` (Apache-2.0). The student weights are derived from SD3-medium and are subject to the license of [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium); see its model card.

## Citation

```bibtex
@article{d2l,
  title   = {Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth},
  journal = {arXiv preprint arXiv:<ARXIV_ID>},
  year    = {<YEAR>}
}
```
