# Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://winterccc.github.io/dooop)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Distill--to--Loop-yellow)](https://huggingface.co/ShimaRinC/Distill-to-Loop)
[![arXiv](https://img.shields.io/badge/arXiv-2611.31977-b31b1b)](https://arxiv.org/abs/2611.31977)

This repository distills SD3-medium into a looped student whose loop count is chosen at sampling time, and it generates images from the result. It also covers Z-Image (see [zimage/](zimage/)).

## Abstract

Looped transformers reuse one block of shared weights along the depth of the network. Recent looped diffusion transformers are trained from scratch, which repeats the full pretraining cost although strong pretrained non-looped diffusion transformers exist. We propose Distill to Loop (Dooop), which distills a pretrained non-looped diffusion transformer into a looped student whose blocks are initialized from the teacher's. We cast distillation as a fixed-point iteration toward the teacher, in which the output after every loop is trained toward the teacher, and each loop is one step of an ordinary differential equation in depth. As a result, one model runs any number of loops at test time without retraining, and the higher-order ODE solvers of diffusion model sampling can be applied along the depth dimension for better results. Meanwhile, because a looped model produces an output after every loop, we also construct loop guidance between the outputs of different loop counts, which improves the generated samples at little extra compute. On SD3-medium, a looped student with about 37% of the teacher's parameters matches the teacher on average over GenEval, ImageReward, PickScore and HPSv2.1 with two to four loops, and with two loops it uses about 58% of the teacher's compute. And on a four-step SD3-medium, loop guidance raises the average of the four metrics from 88.2 to 99.2. This repository covers the SD3-medium and Z-Image distillations and adjustable-loop sampling; loop guidance and the higher-order depth solvers are not included.

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
torchrun --nproc_per_node=2 scripts/prepare_coco.py --images-root /path/to/coco/train2017 --annotations /path/to/coco/annotations/captions_train2017.json --out /path/to/coco2017_512/train
```

### 3. Train

The reported setting uses 2 NVIDIA B200 GPUs with 72 samples per GPU, for 144 samples per step.

```bash
GPUS=0,1 CONFIG=configs/sd3m_dooop.yaml OVERRIDE="data.train_dir=/path/to/coco2017_512/train" bash scripts/launch_train.sh
```

`OVERRIDE` takes space-separated `key=value` overrides of the config, which is how to shorten a run or shrink the batch for a smoke test:

```bash
GPUS=0 OVERRIDE="data.train_dir=/path/to/coco2017_512/train train.max_steps=40 train.per_gpu_batch_size=24" bash scripts/launch_train.sh
```

### 4. Generate

```bash
python scripts/export_student.py --checkpoint /path/to/step_00100000.pt --out /path/to/exported
python scripts/generate.py --student /path/to/dooop-sd3-medium --prompts prompts.txt --out out/ --K 4
```

`--K` sets the loop count at sampling time. The four-step `dooop-sd3-medium-turbo` is sampled with `--sigmas 1.0 0.75 0.5 0.25 --cfg 1.0`.

### 5. Inference in Python

```python
import torch
from huggingface_hub import snapshot_download
from distill_to_loop import load_pipeline

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
GPUS=0,1 CONFIG=configs/sd3m_dooop.yaml RESUME_FROM=/path/to/checkpoints/<name>/step_<N>.pt \
OVERRIDE="data.train_dir=/path/to/coco2017_512/train" bash scripts/launch_train.sh
```

The launch must use the same config and number of GPUs as the run that wrote the checkpoint. Resuming into a directory that already holds later checkpoints or another run's checkpoints is refused.

## 🔗 Related Links

The model weights are at https://huggingface.co/ShimaRinC/Distill-to-Loop, the teacher model is at https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers, the project page is at https://winterccc.github.io/dooop, and the paper is at https://arxiv.org/abs/2611.31977.

## 📄 License

This project is licensed under the Apache-2.0 License (see [LICENSE](LICENSE)).

Third-party components distributed with this project are licensed under their own terms (see [NOTICE](NOTICE)).

The model weights on Hugging Face are derived from SD3-medium and are distributed separately under the Stability AI Non-Commercial Research Community License; see https://huggingface.co/ShimaRinC/Distill-to-Loop.

## 📚 Citation

```bibtex
@article{dooop,
  author  = {Wang, Chenru and An, Shengwei},
  title   = {Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth},
  journal = {arXiv preprint arXiv:2611.31977},
  year    = {2026}
}
```

## 📭 Contact

For questions or issues, please open a GitHub issue in this repository.
