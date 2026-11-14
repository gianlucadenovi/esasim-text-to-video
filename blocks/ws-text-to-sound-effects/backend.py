# ── EsaSim block ──────────────────────────────────────────────────────
# Text to sound effects (ws-text-to-sound-effects) · version 9 · machine block, Python: def run(ctx)
# Generate multiple seeded WAV sound effects from a text prompt using Hugging Face diffusion audio models.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-14
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
#   candidates  number    Candidates per take (0=automatic) = 0
#   negative    text      Negative prompt = "low quality, distorted, speech, music"
#   token       secret    Hugging Face token = (environment variable ESASIM_SECRET_TOKEN)
#   start_from_view bool      Generate requested by Studio (View only) = false
# Runs on  NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch[cuda], diffusers==0.32.2, transformers==4.47.1, huggingface_hub>=0.26,<1.0, tokenizers>=0.21,<0.22, accelerate, safetensors, soundfile, numpy, scipy, torchsde, sentencepiece, protobuf, psutil (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Text to sound effects: an open audio diffusion model from Hugging Face (AudioLDM, AudioLDM 2, Stable Audio Open),
downloaded once into this machine's EsaSim cache, turns one prompt into several seeded WAV takes.

Why the package versions are pinned: with transformers 5.x the AudioLDM / AudioLDM 2 vocoder (SpeechT5HifiGan) is
loaded from a config of another model type and the pipelines produce clipped, NaN-padded broadband noise instead of
the requested sound. diffusers 0.32.2 + transformers 4.47.1 is the last combination in which all three pipelines work
as published (AudioLDM 2 with its original GPT2Model language model)."""
import csv
import inspect
import json
import os
import random
import threading
import time

PRESETS = {
    # name: (repo, family, fallback sample rate)
    "AudioLDM 2": ("cvssp/audioldm2", "audioldm2", 16000),
    "AudioLDM 2 Large": ("cvssp/audioldm2-large", "audioldm2", 16000),
    "AudioLDM": ("cvssp/audioldm-l-full", "audioldm", 16000),
    "Stable Audio Open 1.0": ("stabilityai/stable-audio-open-1.0", "stable", 44100),
}
# (steps on an NVIDIA GPU, steps on a CPU, guidance) from the diffusers model docs
FAMILY = {
    "audioldm": (100, 50, 2.5),
    "audioldm2": (150, 50, 3.5),
    "stable": (100, 50, 7.0),
}
MAX_SECONDS = {"audioldm": 10.0, "audioldm2": 10.0, "stable": 47.0}


def cache_dir(sub):
    base = os.environ.get("ESASIM_DATA_DIR")
    if not base:
        here = os.path.abspath(os.getcwd())
        jobs = os.path.dirname(os.path.dirname(here))
        base = os.path.dirname(jobs) if os.path.basename(jobs) == "jobs" else here
    d = os.path.join(base, "cache", sub)
    os.makedirs(d, exist_ok=True)
    return d


def run(ctx):
    p = ctx.params
    # A normal Blueprint Run prepares the View but never generates audio: only the View's Generate button does.
    if not p.get("start_from_view", False):
        ctx.log("Sound Studio ready. No inference was requested. Open the View and click Generate sounds.")
        ctx.telemetry("studio_status", {"phase": "idle", "message": "Ready. Enter a prompt and click Generate sounds.", "percent": 0})
        ctx.progress(100, "Ready — waiting for Generate sounds in the View")
        return {"status": "idle", "generated": 0}

    # The model cache: the EsaSim data folder, unless that folder is synced by OneDrive (a service account cannot
    # read files OneDrive has turned into online-only placeholders: the weights then look corrupted). Then the
    # account's own Hugging Face cache.
    data = cache_dir("huggingface")
    if "onedrive" not in data.lower():
        os.environ.setdefault("HF_HOME", data)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    hb = {"phase": "preparing", "message": "Preparing", "percent": 1}
    hb_stop = threading.Event()

    def announce(phase, message, percent=None, **extra):
        if percent is not None:
            hb["percent"] = int(percent)
            ctx.progress(int(percent), message)
        hb.update(phase=phase, message=message)
        payload = {"phase": phase, "message": message, "percent": hb["percent"]}
        payload.update(extra)
        ctx.log("[studio] " + message)
        ctx.telemetry("studio_status", payload)

    def heartbeat():
        while not hb_stop.wait(15):
            try:
                ctx.progress(int(hb["percent"]), hb["message"])
                ctx.telemetry("studio_status", {"phase": hb["phase"], "message": hb["message"] + " — still working", "percent": hb["percent"], "heartbeat": True})
            except Exception:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        return generate_all(ctx, p, announce, hb)
    finally:
        hb_stop.set()


def generate_all(ctx, p, announce, hb):
    announce("preparing", "Preparing audio model and checking the execution device…", 1)
    import numpy as np
    import soundfile as sf
    import torch
    import diffusers
    import transformers
    from diffusers import AudioLDM2Pipeline, AudioLDMPipeline, StableAudioPipeline

    # ---- which model
    preset = str(p.get("model") or "AudioLDM 2")
    if preset == "Other Diffusers repository":
        repo = str(p.get("model_id") or "").strip()
        if not repo:
            raise ValueError("Specify the Hugging Face model ID for the custom Diffusers audio model.")
        family = {"audioldm2": "audioldm2", "stableaudio": "stable", "audioldm": "audioldm"}.get(
            str(p.get("custom_family") or "AudioLDM 2").lower().replace(" ", ""))
        if not family:
            raise ValueError("Custom pipeline family must be AudioLDM 2, Stable Audio or AudioLDM.")
        fallback_rate = 44100 if family == "stable" else 16000
    elif preset in PRESETS:
        repo, family, fallback_rate = PRESETS[preset]
    else:
        raise ValueError("Unknown model %r." % preset)
    pipe_cls = {"audioldm": AudioLDMPipeline, "audioldm2": AudioLDM2Pipeline, "stable": StableAudioPipeline}[family]

    # ---- where it runs
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
        where = torch.cuda.get_device_name(0)
    else:
        device, dtype = "cpu", torch.float32
        # Leave cores free: a CPU saturated by PyTorch starves the EsaSim worker, which then stops reporting
        # and the run is marked lost after 180 s.
        threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        torch.set_num_threads(threads)
        try:
            import psutil
            proc = psutil.Process()
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except Exception:
            pass
        where = "CPU (%d threads)" % threads
        ctx.log("WARNING: no CUDA GPU usable from PyTorch %s here: generating on the CPU, which is much slower. "
                "Run on the machine with the NVIDIA GPU for normal speed." % torch.__version__)
    ctx.log("diffusers %s, transformers %s, torch %s, device %s" % (diffusers.__version__, transformers.__version__, torch.__version__, where))

    # ---- settings
    prompt = str(p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Enter a sound description.")
    count = max(1, min(100, int(p.get("count") or 1)))
    seconds = max(1.0, min(MAX_SECONDS[family], float(p.get("seconds") or 8)))
    if float(p.get("seconds") or 8) > MAX_SECONDS[family]:
        ctx.log("Duration capped at %g s for this model." % MAX_SECONDS[family])
    gpu_steps, cpu_steps, default_guidance = FAMILY[family]
    steps = int(p.get("steps") or 0) or (gpu_steps if device == "cuda" else cpu_steps)
    steps = max(1, min(200, steps))
    guidance = float(p.get("guidance") or 0) or default_guidance
    negative = str(p.get("negative") or "").strip()
    seed0 = int(p.get("seed") or 0)
    if seed0 <= 0:
        seed0 = random.SystemRandom().randint(1, 2 ** 30)
    cand = int(p.get("candidates") or 0)
    if family == "stable":
        cand = 1                                   # Stable Audio has no CLAP model to rank takes with
    elif cand <= 0:
        cand = 3 if device == "cuda" else 1
    cand = max(1, min(8, cand))

    # ---- token (block secret, else the runner's profile token, else the environment)
    token = None
    if p.get("token_env"):
        token = os.environ.get(str(p.get("token_env"))) or None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if family == "stable" and not token:
        announce("auth_required", "Stable Audio Open is gated: accept its terms on Hugging Face and save your token "
                 "(Profile > Keys, or the generator block's Hugging Face token).", 2)
        raise ValueError("Stable Audio Open 1.0 needs a Hugging Face token: accept the licence on "
                         "https://huggingface.co/stabilityai/stable-audio-open-1.0, then save your token in your EsaSim "
                         "profile (Profile > Keys) or in the generator block's Hugging Face token setting.")

    # ---- load
    announce("loading", "Downloading / loading %s (the first time downloads it into the machine's cache)…" % repo, 2, model=repo, device=device)
    t_load = time.monotonic()
    kwargs = {"torch_dtype": dtype}
    if token:
        kwargs["token"] = token
    if family == "audioldm2":
        # Newer diffusers expect GPT2LMHeadModel for AudioLDM 2's language model; older ones GPT2Model (as the repo
        # ships it). Give the class this diffusers asks for.
        ann = inspect.signature(AudioLDM2Pipeline.__init__).parameters["language_model"].annotation
        if "LMHead" in getattr(ann, "__name__", str(ann)):
            from transformers import GPT2LMHeadModel
            kwargs["language_model"] = GPT2LMHeadModel.from_pretrained(repo, subfolder="language_model", torch_dtype=dtype, token=token)
    try:
        try:
            pipe = pipe_cls.from_pretrained(repo, **kwargs)
        except OSError as exc:
            if "Unable to load weights" not in str(exc) and "corrupted" not in str(exc):
                raise
            # A weight file in the cache was cut short (an interrupted download or copy): fetch it again once.
            ctx.log("A cached weight file of %s is damaged: downloading the model again." % repo)
            pipe = pipe_cls.from_pretrained(repo, force_download=True, **kwargs)
    except Exception as exc:
        text = str(exc)
        if "401" in text or "403" in text or "gated" in text.lower() or type(exc).__name__ == "GatedRepoError":
            announce("auth_required", "Hugging Face denied access to %s: accept its terms with the account whose token is saved." % repo, 2)
            raise ValueError("Hugging Face authorization failed for %s. Accept the model terms on huggingface.co with the "
                             "account whose token is saved in EsaSim." % repo) from None
        raise
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if family == "stable":
        sample_rate = int(getattr(getattr(pipe, "vae", None), "sampling_rate", 0) or fallback_rate)
    else:
        sample_rate = int(getattr(getattr(pipe, "vocoder", None), "config", None).sampling_rate or fallback_rate)
    loaded = round(time.monotonic() - t_load, 1)
    announce("model_ready", "Model loaded in %.0f s on %s. %d take%s, %g s each, %d steps, guidance %g%s." % (
        loaded, where, count, "s" if count > 1 else "", seconds, steps, guidance,
        ", best of %d candidates per take (CLAP ranking)" % cand if cand > 1 else ""), 5, device=device)

    sig = set(inspect.signature(pipe.__call__).parameters)
    total_steps = count * steps
    state = {"i": 0}

    def step_progress(step):
        done = state["i"] * steps + min(step + 1, steps)
        pct = 5 + 90 * done / total_steps
        hb["percent"] = int(pct)
        hb["message"] = "Generating take %d of %d — step %d of %d" % (state["i"] + 1, count, min(step + 1, steps), steps)
        if step % 5 == 0 or step + 1 >= steps:
            ctx.progress(pct, hb["message"])

    rows = []
    for i in range(count):
        state["i"] = i
        seed = seed0 + i
        announce("generating", "Generating take %d of %d (seed %d)…" % (i + 1, count, seed), 5 + 90 * i / count, current=i + 1, total=count, device=device)
        t0 = time.monotonic()
        args = {"prompt": prompt, "num_inference_steps": steps, "guidance_scale": guidance,
                "generator": torch.Generator(device="cpu").manual_seed(seed)}
        if negative:
            args["negative_prompt"] = negative
        if family == "stable":
            args["audio_end_in_s"] = seconds
        else:
            args["audio_length_in_s"] = seconds
            if cand > 1:
                args["num_waveforms_per_prompt"] = cand   # the pipeline ranks them by CLAP text-audio similarity
        if "callback_on_step_end" in sig:
            args["callback_on_step_end"] = lambda _p, step, _t, kw: (step_progress(step), kw)[1]
        elif "callback" in sig:
            args["callback"] = lambda step, _t, _lat: step_progress(step)
            args["callback_steps"] = 1
        with torch.inference_mode():
            out = pipe(**args)
        audio = out.audios[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T                          # (channels, samples) -> (samples, channels)
        audio = audio[: int(round(seconds * sample_rate))]
        bad = int((~np.isfinite(audio)).sum())
        if bad:
            ctx.log("WARNING: %d non-finite samples in take %d: the model diverged (check the package versions)." % (bad, i + 1))
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = audio - audio.mean(axis=0, keepdims=True)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        clipped = float((np.abs(audio) >= 0.999).mean()) if audio.size else 0.0
        if peak > 0:
            audio = audio * min(0.89 / peak, 10.0)       # peak at -1 dBFS, no clipping
        if clipped > 0.01:
            ctx.log("WARNING: %.1f%% of take %d was clipped at full scale: the model output is saturated (likely broken)." % (clipped * 100, i + 1))
        name = "sound_%02d_seed%d.wav" % (i + 1, seed)
        sf.write(ctx.output("sounds", name), audio, sample_rate, subtype="PCM_16")
        elapsed = round(time.monotonic() - t0, 1)
        rows.append({"variation": i + 1, "file": name, "seed": seed, "prompt": prompt, "model": repo,
                     "seconds": round(len(audio) / sample_rate, 2), "rate": sample_rate, "steps": steps,
                     "guidance": guidance, "candidates": cand, "clipped_pct": round(clipped * 100, 2),
                     "made_in_s": elapsed, "device": where})
        ctx.log("%s: %.1f s to make, seed %d" % (name, elapsed, seed))
        ctx.telemetry("progress", {"variation": i + 1, "file": name, "seed": seed, "made_in_s": elapsed})
        announce("generated", "Finished take %d of %d — %s (%.0f s)" % (i + 1, count, name, elapsed), 5 + 90 * (i + 1) / count,
                 current=i + 1, total=count, file=name, device=device)
        del out, audio
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(ctx.output("table", "sounds.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ctx.write_json("metadata", "generation.json", {
        "model": repo, "preset": preset, "family": family, "prompt": prompt, "negative": negative, "variations": count,
        "first_seed": seed0, "seconds": seconds, "steps": steps, "guidance": guidance, "candidates": cand,
        "device": where, "model_load_s": loaded, "versions": {"diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "torch": torch.__version__}, "outputs": rows,
        "license_note": "Review each model's Hugging Face license before commercial use."})
    announce("done", "All %d sound%s are ready to play." % (count, "s" if count > 1 else ""), 100, current=count, total=count)
    return {"variations": count, "model": repo, "device": where}
