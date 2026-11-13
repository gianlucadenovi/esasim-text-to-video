<div align="center">

# Text to video

**Prompts become short videos (up to 10 seconds), made on your own machine by open models from Hugging Face: watch each one whole as soon as it is made, keep the good ones, and browse them.**

The worker downloads a text-to-video model from Hugging Face once and runs it on the machine's NVIDIA GPU. It turns a prompt into as many variants as you ask for, or a list of prompts into videos, and streams each finished video, whole, to the view: you watch it in full while the next one is being made. Then you decide which to keep and which to remove, and save the kept ones into a dataset that you can browse at any time. Models go from small (1.3 billion parameters, an 8 GB GPU) to the best open one (Wan 2.2 A14B, for cloud GPUs).

![type](https://img.shields.io/badge/type-Generative%20AI-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20GPU-76b900) ![models](https://img.shields.io/badge/models-Hugging%20Face-yellow) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

A video diffusion model starts from random noise covering every frame at once and removes it step by step, guided by the prompt, until a moving scene is left. The noise comes from a **seed**: each seed gives a different video of the same prompt, and the same seed gives the same video again. So one prompt gives as many variants as you ask for, and any one can be made again.

Each video is encoded as an MP4 (H.264, small enough to play in the dashboard) and **sent live, whole, as soon as it is made**: the Video review view plays it in full while the model works on the next one, so you judge each video by watching all of it, not a still. The files, a still of each, a table of what made each video and a summary are the run's results.

The model is downloaded from Hugging Face the first time, only the parts the pipeline needs, into the worker's cache on the machine, and read from there on later runs. Nothing is sent to an online service: the prompts and the videos stay on your machine and in your EsaSim workspace.

## Flows

- **1 · One prompt, several videos**: Wan 2.1 (1.3B) makes 4 videos of a fox in the snow (seeds 1 to 4, 5 seconds each). Set **Videos per prompt** to any number.
- **2 · A list of prompts**: four different prompts (one per line), two variants each.
- **3 · Small to large: the models**: the same prompt and seeds on Wan 2.1 1.3B, LTX-Video 2B, Kandinsky 5.0 Video Lite 2B and Wan 2.2 TI2V 5B: their videos side by side, and a chart of the time each takes per video.
- **4 · Browse the kept videos**: the dataset where you saved the videos you kept, in the same review view: watch them, and change your mind.

## Review: keep and remove

The **Video review** view shows each video as a card; it plays when the pointer is over it. Click it to watch it large, whole: the arrows (or the arrow keys, or a swipe) go through all of them.

- **K** keep, **R** remove, **U** undo, **Space** pause, **Esc** back to the grid; the same with the buttons.
- The filters show the videos **to decide**, **kept** or **removed**, with their counts.
- Decisions stay on this computer, for each run.
- **Save the kept videos** puts them, with their stills and `kept.csv` (prompt, seed, model of each), into a dataset of your choice, in a folder, as one new version; or downloads them as one zip. Flow 4 then shows that dataset.
- **Save choices** gives `choices.csv`: every video with its decision.

Run results themselves are never deleted: removing a video keeps it out of your kept set.

## The models

| Model | Size | Video | Length | Licence | GPU |
|---|---|---|---|---|---|
| **Wan 2.1 T2V** (Alibaba), the default | 1.3B | 832×480, 16 fps | 5 s (up to 10) | Apache-2.0 | 8 GB |
| **LTX-Video 0.9.5** (Lightricks), the fastest | 2B | 768×512, 24 fps | 5 s (up to 10) | Open RAIL-M, with use restrictions | 8 GB |
| **Kandinsky 5.0 Video Lite** (Kandinsky Lab), distilled, 16 steps | 2B | 768×512, 24 fps | 5 or 10 s (a model for each) | MIT | 12 GB (8 GB slowly) |
| **Wan 2.2 TI2V** (Alibaba) | 5B | 1280×704, 24 fps | 5 s (up to 10) | Apache-2.0 | 24 GB (8 GB very slowly) |
| **HunyuanVideo 1.5** (Tencent) | 8.3B | 848×480, 24 fps | 5 s (up to 10) | Tencent Hunyuan Community: not for the EU, the UK or South Korea | 14 GB and more |
| **Wan 2.2 T2V A14B** (Alibaba), the best open model | 27B (14B at a time) | 1280×720, 16 fps | 5 s (up to 10) | Apache-2.0 | 80 GB, and 96 GB of computer memory: a cloud machine |

**Seconds** sets the length of every video, at most 10 s (0: the model's own, 5 s). Beyond what a model was trained on (Wan: 5 s), motion may slow down or repeat. **Width** and **Height** change the size (swap them for a vertical video). Any other text-to-video model in the diffusers format can be named instead (Model: Other), from Hugging Face or a folder on the machine.

## Where it runs

- **NVIDIA GPU**: the models compute in bfloat16. A model that fits the GPU runs on it whole; a little too large, its parts take turns on it; larger than that (the text encoders of these models alone are larger than 8 GB), its layers wait in the computer's memory and are streamed to the GPU group by group as they are needed, the next group copied while the current one computes. It works on an 8 GB GPU, more slowly than on one that holds the whole model; the log says which way each model runs. If the GPU runs out of memory, the block goes on with the next lighter way by itself.
- **CPU**: only for a test (Accelerator: CPU): a short video takes hours. Tenstorrent cards cannot run these video models yet.

## What you need

- A machine with the EsaSim worker and an NVIDIA GPU (Project › Compute). With an 8 GB GPU, 32 GB of computer memory or more is best (the models wait there).
- Disk space for each model, downloaded the first time: tens of gigabytes per model, most of it the text encoder. The block says the size before downloading and checks the free space first. Flow 3 downloads four models.
- Internet access the first time.
- Python packages, installed by the worker the first time: `torch` (with CUDA), `diffusers`, `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `ftfy`, `huggingface_hub`, `imageio-ffmpeg`, `psutil`, `pillow`, `numpy`.
- No account or key. A Hugging Face token saved once in your EsaSim profile (**Profile › Keys**) gives higher download limits and opens gated models.

## How to run it

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Open a flow under **Blueprint**; change the **Prompts** (one per line) and **Videos per prompt** of the Text to video block if you like.
3. Press **Run**. The first run downloads the model (the progress shows on the block); then the videos come one after the other, with the time left on the block.
4. Open the **View**: each video plays there, whole, as soon as it is made. Keep or remove them, then **Save the kept videos**.

## Settings worth knowing

- **Prompts**: one per line: the subject and what it does, the camera, the light, the style. **Negative prompt**: what the videos should not show (empty: the model's own).
- **Videos per prompt**: variant k uses seed **First seed** + k − 1; 0 gives new videos at every run.
- **Seconds**, **Width**, **Height**, **Steps**, **Guidance**: 0 keeps the model's own.
- **File size**: small (each MP4 under 4.5 MB, so it is uploaded to EsaSim) or high (larger files that stay on the machine and are fetched when opened).
- **Stream live**: whole videos (the default), or stills only.

## What you get

- **Videos**: `video_01_seed1.mp4` and its still `video_01_seed1.jpg`, … (with several prompts `video_01_p1_seed1.mp4`), and `generation.json` (model, licence, settings, time) to make the same videos again.
- **Videos table** `videos.csv`: for each video its prompt, seed, length, frames, size, steps, guidance, the seconds it took, model, licence and device.
- **Summary** `summary.csv`: one row per run (model, size, length, steps, device, memory, seconds per video, loading time, licence); flow 3 joins those of four models into a chart.

## Models and licences

- **Wan 2.1 T2V 1.3B**, [huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), **Wan 2.2 TI2V 5B**, [huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), and **Wan 2.2 T2V A14B**, [huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers): Apache-2.0.
- **LTX-Video 0.9.5**, [huggingface.co/Lightricks/LTX-Video-0.9.5](https://huggingface.co/Lightricks/LTX-Video-0.9.5): Open RAIL-M (Lightricks), which allows commercial use, with use restrictions listed in the licence.
- **Kandinsky 5.0 Video Lite**, [huggingface.co/kandinskylab](https://huggingface.co/kandinskylab) (T2V Lite distilled, 16 steps, 5 s and 10 s): MIT.
- **HunyuanVideo 1.5**, [huggingface.co/tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) (Diffusers weights by hunyuanvideo-community): Tencent Hunyuan Community licence, which does not apply in the European Union, the United Kingdom and South Korea.
- With another model, the block reads the licence from its model card, records it with every video, and says so in the log when it limits use or is not stated.
- The code of the blocks is EsaSim's.

## Notes

- The videos are not filtered: what they show comes from the prompt. Use the project responsibly, and check the licence of a model before using its videos.
- The first video of a run takes longer (loading and warming up); the Summary gives it apart.
- Same seed, same prompt, same settings, same model: the same video, on any machine with the same kind of GPU.
- Results up to 5 MB each (25 MB per block) are uploaded to EsaSim; larger ones stay on the machine and are fetched when opened. The live stream carries every video whole regardless, while the run is shown.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (generative-ai).

### Blueprints in this repository

- [1 · One prompt, several videos](blueprints/1-one-prompt-several-videos.json) · 4 blocks · version 4
- [2 · A list of prompts](blueprints/2-a-list-of-prompts.json) · 4 blocks
- [3 · Small to large: the models](blueprints/3-small-to-large-the-models.json) · 10 blocks
- [4 · Browse the kept videos](blueprints/4-browse-the-kept-videos.json) · 2 blocks
- [5 · Text to sound — multiple variations](blueprints/5-text-to-sound-multiple-variations.json) · 4 blocks · version 1

### Workspace blocks it uses

- [Text to sound effects](blocks/ws-text-to-sound-effects) · version 1

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
