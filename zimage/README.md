# Dooop for Z-Image

This folder distills [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) into a Dooop looped student with 3 pre blocks, 6 shared blocks repeated K times, and 3 post blocks. The student is trained at K = 4, and the loop count K is chosen at sampling time. The method and the paper are described in the [top-level README](../README.md). All commands below run from the repository root.

## 🚀 Getting Started

### 1. Install

Use the same environment as the SD3-medium part (`pip install -r requirements.txt`, see the [top-level README](../README.md)). The Z-Image code needs no other package.

### 2. Prepare the teacher and data

The Z-Image teacher is Tongyi-MAI/Z-Image (Apache-2.0). The code reads it from the local Hugging Face cache at the revision pinned in `zimage/configs/zimage_dooop.yaml` and does not download it, so download it first:

```bash
hf download Tongyi-MAI/Z-Image --revision 04cc4abb7c5069926f75c9bfde9ef43d49423021
```

To use another cache, set `HF_HUB_CACHE`, which every script reads. Training takes the override `paths.hf_cache=...` in `OVERRIDE`, and `prepare_coco.py`, `export_student.py` and `generate.py` also take `--hf-cache`.

Training reads COCO 2017 train at 512 px as pre-encoded shards. The following command prepares them with 2 GPUs:

```bash
torchrun --nproc_per_node=2 zimage/scripts/prepare_coco.py --images-root /path/to/coco/train2017 --annotations /path/to/coco/annotations/captions_train2017.json --out /path/to/zimage_coco2017_512/train
```

### 3. Train

The released recipe uses 2 NVIDIA H200 GPUs with 48 samples per GPU, for 96 samples per step, and it used 133 to 135 GB of memory per GPU.

```bash
GPUS=0,1 CONFIG=zimage/configs/zimage_dooop.yaml OVERRIDE="data.train_dir=/path/to/zimage_coco2017_512/train" bash zimage/scripts/launch_train.sh
```

`OVERRIDE` takes space-separated `key=value` overrides of the config. Checkpoints are written to `<output_dir>/<name>/`, and `train.keep_last` of them are kept.

### 4. Export

```bash
python zimage/scripts/export_student.py --checkpoint /path/to/step_00020000.pt --out /path/to/exported
```

The export holds the EMA weights of the checkpoint.

### 5. Generate

```bash
python zimage/scripts/generate.py --student /path/to/exported --prompts prompts.txt --out out/ --K 4
```

`--prompts` is a text file with one prompt per line, or a `.jsonl` file with a `prompt` field. `--K` sets the loop count at sampling time. The defaults are 512 px, 28 steps and guidance scale g = 8.0, which is the guidance scale of the Z-Image pipeline (the CFG weight is w = g + 1 = 9).

### 6. Inference in Python

```python
import sys
sys.path[:0] = ["zimage", "."]

from dooop_zimage import encode_prompts, generate_images, load_student, load_vae_and_scheduler_factory

device = "cuda"
student = load_student("/path/to/exported", device=device)
vae, scheduler_factory = load_vae_and_scheduler_factory(device)
encodings = encode_prompts(["a lighthouse on a cliff at sunset"], device)
(_, _, image), = generate_images(student, vae, scheduler_factory, encodings, loop_rounds=4, steps=28, guidance_scale=8.0,
                                 resolution=512, seed=0, micro_batch_size=1, device=device)
image.save("out.png")
```

The loop count is set with `loop_rounds`.

## 🔁 Resuming a Run

```bash
GPUS=0,1 CONFIG=zimage/configs/zimage_dooop.yaml RESUME_FROM=/path/to/checkpoints/<name>/step_<N>.pt \
OVERRIDE="data.train_dir=/path/to/zimage_coco2017_512/train" bash zimage/scripts/launch_train.sh
```

The launch must use the same config, number of GPUs and rows per GPU as the run that wrote the checkpoint. A run directory that holds a checkpoint is continued only with `RESUME_FROM`.

## 📄 License

This project is licensed under the Apache-2.0 License (see [LICENSE](../LICENSE)). The teacher Tongyi-MAI/Z-Image is licensed under Apache-2.0. Third-party components are listed in [NOTICE](../NOTICE).
