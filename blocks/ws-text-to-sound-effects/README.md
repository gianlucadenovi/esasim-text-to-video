# Text to sound effects

Generate multiple seeded WAV sound effects from a text prompt using Hugging Face diffusion audio models.

Pick a model, describe the sound and set the number of takes and their length. Each take is a WAV file, with a CSV and JSON manifest. AudioLDM and AudioLDM 2 make several candidates per take and keep the one that best matches the prompt (CLAP ranking): Candidates 0 = 3 on a GPU, 1 on a CPU. Stable Audio Open is gated: accept its licence on Hugging Face and save your token in Profile > Keys (or in this block). AudioLDM lengths are capped at 10 s, Stable Audio at 47 s. Runs best on an NVIDIA GPU; on a CPU expect minutes per take. Model licences may restrict commercial use.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-text-to-sound-effects` · version 6 |
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
| `steps` Inference steps (0=automatic) | number | `0` | 0: AudioLDM 100, AudioLDM 2 150, Stable Audio 100 on a GPU; 50 on a CPU |
| `guidance` Guidance (0=automatic) | number | `0` | 0: AudioLDM 2.5, AudioLDM 2 3.5, Stable Audio 7 |
| `candidates` Candidates per take (0=automatic) | number | `0` | AudioLDM / AudioLDM 2 make this many and keep the one closest to the prompt (CLAP). 0: 3 on a GPU, 1 on a CPU |
| `negative` Negative prompt | text | `"low quality, distorted, speech, music"` |  |
| `token` Hugging Face token | secret | (secret) | Empty: the Hugging Face token in the profile of whoever runs the flow. Needed for gated models (Stable Audio Open) |
| `start_from_view` Generate requested by Studio (View only) | bool | `false` | Leave off. The View enables this only when Generate sounds is clicked; a normal Blueprint Run does not start inference. |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
