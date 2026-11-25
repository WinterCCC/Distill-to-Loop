# Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://winterccc.github.io/dooop)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Distill--to--Loop-yellow)](https://huggingface.co/ShimaRinC/Distill-to-Loop)
[![arXiv](https://img.shields.io/badge/arXiv-2611.31977-b31b1b)](https://arxiv.org/abs/2611.31977)

This repository distills SD3-medium (see [sd3/README.md](sd3/README.md)) and Z-Image (see [zimage/README.md](zimage/README.md)) into looped students whose loop count is chosen at sampling time, and it generates images from the results.

## Abstract

Looped transformers reuse one block of shared weights along the depth of the network. Recent looped diffusion transformers are trained from scratch, which repeats the full pretraining cost although strong pretrained non-looped diffusion transformers exist. We propose Distill to Loop (Dooop), which distills a pretrained non-looped diffusion transformer into a looped student whose blocks are initialized from the teacher's. We cast distillation as a fixed-point iteration toward the teacher, in which the output after every loop is trained toward the teacher, and each loop is one step of an ordinary differential equation in depth. As a result, one model runs any number of loops at test time without retraining, and the higher-order ODE solvers of diffusion model sampling can be applied along the depth dimension for better results. Meanwhile, because a looped model produces an output after every loop, we also construct loop guidance between the outputs of different loop counts, which improves the generated samples at little extra compute. On SD3-medium, a looped student with about 37% of the teacher's parameters matches the teacher on average over GenEval, ImageReward, PickScore and HPSv2.1 with two to four loops, and with two loops it uses about 58% of the teacher's compute. And on a four-step SD3-medium, loop guidance raises the average of the four metrics from 88.2 to 99.2. This repository covers the SD3-medium and Z-Image distillations and adjustable-loop sampling; loop guidance and the higher-order depth solvers are not included.

## 📁 Repository layout

- `dooop/`: the looped student with its training and sampling code (the SD3-medium implementation; Z-Image reuses its shared parts).
- `sd3/`: the SD3-medium configs, scripts and tests (see [sd3/README.md](sd3/README.md)).
- `zimage/`: the Z-Image package, configs, scripts and tests (see [zimage/README.md](zimage/README.md)).

Both parts use the same environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
