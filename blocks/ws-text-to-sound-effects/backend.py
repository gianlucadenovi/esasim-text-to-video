# ── EsaSim block ──────────────────────────────────────────────────────
# Text to sound effects (ws-text-to-sound-effects) · version 1 · machine block, Python: def run(ctx)
# Generate multiple seeded WAV sound effects from a text prompt using Hugging Face diffusion audio models.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-13
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   (none)
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   sounds      Files (wav)               Generated WAV files
#   table       Table (csv)               Generation details
#   metadata    Files (json)              Generation manifest
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   model       select    Hugging Face audio model = "AudioLDM 2" of AudioLDM 2 | AudioLDM 2 Large | Stable Audio Open 1.0 | …
#   model_id    text      Custom model ID = ""
#   custom_family select    Custom pipeline family = "AudioLDM 2" of AudioLDM 2 | Stable Audio | AudioLDM
#   prompt      textarea  Sound description = "A sudden powerful thunderclap directly overhead, followed by a long low rollin…
#   count       number    Number of variations = 4
#   seconds     number    Duration per clip (seconds) = 8
#   seed        number    First random seed (0=random) = 1
#   steps       number    Inference steps (0=automatic) = 0
#   guidance    number    Guidance (0=automatic) = 0
#   negative    text      Negative prompt = "speech, music"
#   token       secret    Hugging Face access token = (environment variable ESASIM_SECRET_TOKEN)
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

import csv
import json
import os
import random
import time

def run(ctx):
    import torch
    import numpy as np
    import soundfile as sf
    from diffusers import StableAudioPipeline, AudioLDM2Pipeline, AudioLDMPipeline

    p = ctx.params
    preset = str(p.get("model", "AudioLDM 2"))
    custom = str(p.get("model_id", "") or "").strip()
    repos = {
        "Stable Audio Open 1.0": ("stabilityai/stable-audio-open-1.0", "stable", 44100),
        "AudioLDM 2": ("cvssp/audioldm2", "audioldm2", 16000),
        "AudioLDM 2 Large": ("cvssp/audioldm2-large", "audioldm2", 16000),
        "AudioLDM": ("cvssp/audioldm", "audioldm", 16000),
    }
    if preset == "Other Diffusers repository":
        if not custom:
            raise ValueError("Specify the Hugging Face model ID for the custom Diffusers audio model.")
        repo, family, rate = custom, str(p.get("custom_family", "AudioLDM 2")).lower().replace(" ", ""), 16000
    else:
        repo, family, rate = repos[preset]
    prompt = str(p.get("prompt", "") or "").strip()
    if not prompt:
        raise ValueError("Enter a sound description.")
    count = max(1, min(100, int(p.get("count", 4))))
    seconds = max(1., min(47., float(p.get("seconds", 8))))
    steps = max(1, min(200, int(p.get("steps", 0) or (100 if family == "stable" else 50))))
    seed0 = int(p.get("seed", 1))
    if seed0 == 0:
        seed0 = random.SystemRandom().randint(1, 2**30)
    negative = str(p.get("negative", "") or "").strip()
    guidance = float(p.get("guidance", 0) or (7. if family == "stable" else 2.5))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    token_env = p.get("token_env")
    token = os.environ.get(token_env) if token_env else None
    if family == "stable":
        pipe_cls = StableAudioPipeline
        seconds = min(seconds, 47.)
    elif family == "audioldm2":
        pipe_cls = AudioLDM2Pipeline
        seconds = min(seconds, 10.)
    elif family == "audioldm":
        pipe_cls = AudioLDMPipeline
        seconds = min(seconds, 10.)
    else:
        raise ValueError("Custom model family must be Stable Audio, AudioLDM 2, or AudioLDM")
    ctx.progress(1, "Loading " + repo)
    kwargs = {"torch_dtype": dtype}
    if token:
        kwargs["token"] = token
    start = time.monotonic()
    pipe = pipe_cls.from_pretrained(repo, **kwargs)
    if device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    loaded = round(time.monotonic() - start, 2)
    sample_rate = int(getattr(pipe, "sample_rate", 0) or rate)
    rows = []
    ctx.log("Generating %d sound variations with %s on %s" % (count, repo, device))
    for i in range(count):
        seed = seed0 + i
        generator = torch.Generator(device=device).manual_seed(seed)
        ctx.progress(5 + int(90 * i / count), "Generating variation %d/%d" % (i + 1, count))
        t0 = time.monotonic()
        args = {"prompt": prompt, "num_inference_steps": steps, "generator": generator, "guidance_scale": guidance}
        if negative:
            args["negative_prompt"] = negative
        if family == "stable":
            args["audio_end_in_s"] = seconds
        else:
            args["audio_length_in_s"] = seconds
        with torch.inference_mode():
            result = pipe(**args)
        audio = np.asarray(result.audios[0], dtype=np.float32)
        # Output may be [channels, time], or [time, channels].
        if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = np.nan_to_num(audio, nan=0, posinf=0, neginf=0)
        audio = np.clip(audio, -1, 1)
        name = "sound_%02d_seed%d.wav" % (i + 1, seed)
        sf.write(ctx.output("sounds", name), audio, sample_rate, subtype="PCM_16")
        elapsed = round(time.monotonic() - t0, 2)
        rows.append({"variation": i + 1, "file": name, "seed": seed, "prompt": prompt, "model": repo, "seconds": round(len(audio) / sample_rate, 3), "rate": sample_rate, "made_in_s": elapsed, "device": device})
        ctx.log("%s: %.1fs, seed %d" % (name, elapsed, seed))
        ctx.telemetry("progress", {"variation": i + 1, "file": name, "seed": seed, "made_in_s": elapsed})
        ctx.progress(5 + int(90 * (i + 1) / count), "Created %d/%d WAV files" % (i + 1, count))
        del result, audio
        if device == "cuda":
            torch.cuda.empty_cache()
    f = ctx.output("table", "sounds.csv")
    with open(f, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ctx.write_json("metadata", "generation.json", {"model": repo, "preset": preset, "prompt": prompt, "variations": count, "first_seed": seed0, "seconds_requested": seconds, "steps": steps, "guidance": guidance, "negative": negative, "device": device, "model_load_s": loaded, "outputs": rows, "license_note": "Review each model's Hugging Face license before commercial use."})
    ctx.progress(100, "%d sound files ready" % count)
    return {"result": {"variations": count, "model": repo}}
