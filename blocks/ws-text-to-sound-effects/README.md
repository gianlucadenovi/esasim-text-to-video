# Text to sound effects

Generate multiple seeded WAV sound effects from a text prompt using Hugging Face diffusion audio models.

Select Stable Audio Open, AudioLDM 2, AudioLDM 2 Large or AudioLDM; describe the sound, set variations and duration. Each result is an individual WAV file, with a CSV and JSON manifest. Stable Audio Open can require accepting model terms and a Hugging Face token. AudioLDM family length is capped at 10 seconds. Model licenses may restrict commercial use.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-text-to-sound-effects` · version 3 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Outputs

| Port | Type | Description |
|---|---|---|
| `sounds` Generated WAV files | Files (wav) |  |
| `table` Generation details | Table (csv) |  |
| `metadata` Generation manifest | Files (json) |  |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `model` Hugging Face audio model | select | `"AudioLDM 2"` |  |
| `model_id` Custom model ID | text | `""` |  |
| `custom_family` Custom pipeline family | select | `"AudioLDM 2"` |  |
| `prompt` Sound description | textarea | `"A sudden powerful thunderclap directly overhead, followed by a long low rolling rumble that slowly fades into a quiet rainstorm. Photorealistic field recording, no music, no voices."` |  |
| `count` Number of variations | number | `4` |  |
| `seconds` Duration per clip (seconds) | number | `8` |  |
| `seed` First random seed (0=random) | number | `1` |  |
| `steps` Inference steps (0=automatic) | number | `0` |  |
| `guidance` Guidance (0=automatic) | number | `0` |  |
| `negative` Negative prompt | text | `"speech, music"` |  |
| `token` Hugging Face access token | secret | (secret) |  |
| `start_from_view` Generate requested by Studio (View only) | bool | `false` | Leave off. The View automatically enables this only when Generate sounds is clicked; a normal Blueprint Run does not start inference. |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
