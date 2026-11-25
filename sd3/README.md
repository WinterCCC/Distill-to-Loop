# Dooop for SD3-medium

This folder distills SD3-medium into a Dooop looped student whose loop count is chosen at sampling time, and it generates images from the result. The method and the paper are described in the [top-level README](../README.md). All commands below run from the repository root.

## 🚀 Getting Started

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare weights and data

| Asset | Link |
|---|---|
| SD3-medium teacher | https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers |
| Dooop student weights | https://huggingface.co/ShimaRinC/Distill-to-Loop |
| COCO 2017 train images and captions | https://cocodataset.org/#download |

SD3-medium is gated. Accept its license on the Hub, log in with `hf auth login`, and download it with `hf download stabilityai/stable-diffusion-3-medium-diffusers`.

Training reads COCO 2017 train at 512 px as pre-encoded shards. The following command prepares them with 2 GPUs:

```bash
torchrun --nproc_per_node=2 sd3/scripts/prepare_coco.py --images-root /path/to/coco/train2017 --annotations /path/to/coco/annotations/captions_train2017.json --out /path/to/coco2017_512/train
```

### 3. Train

The reported setting uses 2 NVIDIA B200 GPUs with 72 samples per GPU, for 144 samples per step.

```bash
GPUS=0,1 CONFIG=sd3/configs/sd3m_dooop.yaml OVERRIDE="data.train_dir=/path/to/coco2017_512/train" bash sd3/scripts/launch_train.sh
```

`OVERRIDE` takes space-separated `key=value` overrides of the config, which is how to shorten a run or shrink the batch for a smoke test:

```bash
GPUS=0 OVERRIDE="data.train_dir=/path/to/coco2017_512/train train.max_steps=40 train.per_gpu_batch_size=24" bash sd3/scripts/launch_train.sh
```

### 4. Generate

```bash
python sd3/scripts/export_student.py --checkpoint /path/to/step_00100000.pt --out /path/to/exported
python sd3/scripts/generate.py --student /path/to/dooop-sd3-medium --prompts prompts.txt --out out/ --K 4
```

`--K` sets the loop count at sampling time. The four-step `dooop-sd3-medium-turbo` is sampled with `--sigmas 1.0 0.75 0.5 0.25 --cfg 1.0`.

### 5. Inference in Python

```python
import torch
from huggingface_hub import snapshot_download
from dooop import load_pipeline

student_dir = snapshot_download("ShimaRinC/Distill-to-Loop") + "/dooop-sd3-medium"
pipe = load_pipeline(student_dir, device="cuda")
pipe.transformer.loop_rounds = 4
image = pipe("a lighthouse on a cliff at sunset", num_inference_steps=28, guidance_scale=7.0, height=512, width=512,
             generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("out.png")
```

The loop count is set with `pipe.transformer.loop_rounds`.

## 🔁 Resuming a Run

```bash
GPUS=0,1 CONFIG=sd3/configs/sd3m_dooop.yaml RESUME_FROM=/path/to/checkpoints/<name>/step_<N>.pt \
OVERRIDE="data.train_dir=/path/to/coco2017_512/train" bash sd3/scripts/launch_train.sh
```

The launch must use the same config and number of GPUs as the run that wrote the checkpoint. Resuming into a directory that already holds later checkpoints or another run's checkpoints is refused.

## 📄 License

This project is licensed under the Apache-2.0 License (see [LICENSE](../LICENSE)). Third-party components are listed in [NOTICE](../NOTICE). The model weights on Hugging Face are derived from SD3-medium and are distributed separately under the Stability AI Non-Commercial Research Community License; see https://huggingface.co/ShimaRinC/Distill-to-Loop.
