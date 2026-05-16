#!/Users/master/lk/.venv/bin/python
"""
Continuous Voice Agent
  STT: Qwen3-ASR (MLX) or faster-whisper — choose at startup (env, flag, or menu)
  LLM: OpenRouter API  — streamed
  TTS: Kokoro (local, free)  — chunked pipeline

Latency optimisation
  LLM tokens → sentence buffer → [synth thread] → audio queue → [play thread]
  Synthesis of sentence N+1 overlaps with playback of sentence N.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python3 voice_agent.py
  python3 voice_agent.py --stt whisper
  python3 voice_agent.py --tts openai
  python3 voice_agent.py --llm cerebras
  VOICE_STT_BACKEND=whisper python3 voice_agent.py

Optional environment (see .env.example):
  VOICE_SILENCE_THRESHOLD, VOICE_SILENCE_SECS, VOICE_MIN_SPEECH_SECS, VOICE_MAX_RECORD_SECS
  VOICE_INPUT_DEVICE (PortAudio index or substring of device name)
  VOICE_LIST_DEVICES=1 — print all input-capable devices at startup
  VOICE_POST_PLAYBACK_SLEEP — seconds to wait after TTS before listening (reduces speaker echo)
  VOICE_MAX_HISTORY_PAIRS, VOICE_MAX_CONTEXT_CHARS, VOICE_MEMORY_NOTES_MAX_CHARS — context window
  VOICE_MEMORY_PATH — JSON file to persist memory + recent turns across runs (empty = off)
  VOICE_LLM_MAX_TOKENS — completion cap for spoken replies
  VOICE_LLM_BACKEND — openrouter | cerebras (also --llm)
  CEREBRAS_API_KEY, CEREBRAS_MODEL, CEREBRAS_BASE_URL — Cerebras Inference (see cerebras.ai)
  HF_TOKEN — Hugging Face read token (optional; higher rate limits for model downloads)
  KOKORO_REPO_ID — Kokoro weights repo on the Hub (default hexgrad/Kokoro-82M)
  Say phrases like “Clear your memory.” / “forget everything” to reset session + disk memory.
  VOICE_STT_BACKEND — qwen | whisper (overridden by --stt; if unset and TTY, menu)
  WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE — faster-whisper options
  VOICE_TTS_BACKEND — kokoro | openai | openvox (also --tts)
  OPENVox_* — local voice API at http://127.0.0.1:8000/v1 (see OPENVox_BASE_URL)
"""

import base64
import io
import json
import os
import queue
import re
import sys
import time
import threading
import warnings

# Torch / Kokoro stack emits noisy, third-party warnings we cannot fix here.
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.rnn",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)

import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Dict, Tuple, Callable, Any, Union, TYPE_CHECKING

SampleRateSpec = Union[int, Callable[[], int]]
import requests

if TYPE_CHECKING:
    from kokoro import KPipeline

# ── Configuration ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
# OpenRouter reasoning: effort=none disables thinking tokens (best for voice latency)
OPENROUTER_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "none").strip().lower()
# Cerebras Inference — OpenAI-compatible chat API (https://inference-docs.cerebras.ai)
CEREBRAS_API_KEY    = os.environ.get("CEREBRAS_API_KEY", "").strip()
CEREBRAS_BASE_URL   = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").strip().rstrip("/")
CEREBRAS_MODEL      = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b").strip()
# gpt-oss / glm stream thinking in delta.reasoning unless reasoning_format=hidden (voice default)
CEREBRAS_REASONING_FORMAT = os.environ.get("CEREBRAS_REASONING_FORMAT", "hidden").strip().lower()
CEREBRAS_REASONING_EFFORT = os.environ.get("CEREBRAS_REASONING_EFFORT", "low").strip().lower()
SYSTEM_PROMPT       = os.environ.get("SYSTEM_PROMPT",
    "You are Jullie, my AI assistant. "
    "Keep responses concise and conversational — 1-3 sentences unless more detail is asked for. "
    "Avoid emojis, markdown, bullet points, or special characters since your output will be spoken aloud."
)

# Audio capture (override with VOICE_* env vars — defaults tuned for quieter mics / shorter utterances)
SAMPLE_RATE         = 16000   # Qwen3-ASR expects 16 kHz
CHANNELS            = 1
BLOCK_SIZE          = 512
SILENCE_THRESHOLD   = float(os.environ.get("VOICE_SILENCE_THRESHOLD", "0.012"))
SILENCE_SECS        = float(os.environ.get("VOICE_SILENCE_SECS", "1.8"))
MIN_SPEECH_SECS     = float(os.environ.get("VOICE_MIN_SPEECH_SECS", "0.25"))
MAX_RECORD_SECS     = float(os.environ.get("VOICE_MAX_RECORD_SECS", "30"))
POST_PLAYBACK_SLEEP = float(os.environ.get("VOICE_POST_PLAYBACK_SLEEP", "0.25"))

# Input device: unset = PortAudio default; int string = index; otherwise substring match on input device name
VOICE_INPUT_DEVICE_RAW = os.environ.get("VOICE_INPUT_DEVICE", "").strip()
VOICE_LIST_DEVICES     = os.environ.get("VOICE_LIST_DEVICES", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

# Qwen3-ASR STT (mlx-qwen3-asr, Apple Silicon) — used when VOICE_STT_BACKEND=qwen
# Options: "Qwen/Qwen3-ASR-0.6B" (fast) | "Qwen/Qwen3-ASR-1.7B" (accurate)
QWEN_MODEL          = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-ASR-0.6B")

# faster-whisper — used when VOICE_STT_BACKEND=whisper
WHISPER_MODEL       = os.environ.get("WHISPER_MODEL", "base.en").strip() or "base.en"
WHISPER_DEVICE      = os.environ.get("WHISPER_DEVICE", "auto").strip() or "auto"
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "default").strip() or "default"
WHISPER_BEAM_SIZE   = max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "1")))

# Kokoro TTS (HF_TOKEN in env improves Hub download limits — see .env.example)
KOKORO_REPO_ID      = os.environ.get("KOKORO_REPO_ID", "hexgrad/Kokoro-82M").strip() or "hexgrad/Kokoro-82M"
KOKORO_LANG         = os.environ.get("KOKORO_LANG", "a").strip() or "a"  # 'a' American, 'b' British
KOKORO_VOICE        = os.environ.get("KOKORO_VOICE", "af_heart").strip() or "af_heart"
KOKORO_SPEED        = float(os.environ.get("KOKORO_SPEED", "1.0"))
KOKORO_SAMPLE_RATE  = int(os.environ.get("KOKORO_SAMPLE_RATE", "24000"))

# OpenAI Speech API (or any OpenAI-compatible POST /audio/speech endpoint)
OPENAI_SPEECH_API_KEY = (
    os.environ.get("OPENAI_SPEECH_API_KEY", "").strip()
    or os.environ.get("OPENAI_API_KEY", "").strip()
)
OPENAI_SPEECH_BASE_URL = (
    os.environ.get("OPENAI_SPEECH_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    or "https://api.openai.com/v1"
)
OPENAI_SPEECH_MODEL   = os.environ.get("OPENAI_SPEECH_MODEL", "tts-1").strip() or "tts-1"
OPENAI_SPEECH_VOICE   = os.environ.get("OPENAI_SPEECH_VOICE", "alloy").strip() or "alloy"
OPENAI_SPEECH_FORMAT  = os.environ.get("OPENAI_SPEECH_FORMAT", "pcm").strip().lower() or "pcm"
OPENAI_SPEECH_SPEED   = float(os.environ.get("OPENAI_SPEECH_SPEED", "1.0"))
OPENAI_SPEECH_SAMPLE_RATE = int(os.environ.get("OPENAI_SPEECH_SAMPLE_RATE", "24000"))

# OpenVox local voice API (OpenAI-style paths under /v1)
OPENVox_BASE_URL = (
    os.environ.get("OPENVox_BASE_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
    or "http://127.0.0.1:8000/v1"
)
OPENVox_MODEL     = os.environ.get("OPENVox_MODEL", "").strip()
OPENVox_LANGUAGE  = os.environ.get("OPENVox_LANGUAGE", "en").strip() or "en"
OPENVox_VOICE     = os.environ.get("OPENVox_VOICE", "").strip()
OPENVox_STREAM    = os.environ.get("OPENVox_STREAM", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)
OPENVox_429_WAIT  = float(os.environ.get("OPENVox_429_WAIT", "2.0"))
OPENVox_MAX_RETRIES = max(1, int(os.environ.get("OPENVox_MAX_RETRIES", "12")))
# active = preload only the selected model; all = POST /models/{id}/load for every model; or comma list
OPENVox_PRELOAD_MODELS = os.environ.get("OPENVox_PRELOAD_MODELS", "active").strip()

# Startup: load STT/TTS weights and optional TTS warm-up synthesis (avoids slow first reply)
VOICE_PRELOAD          = os.environ.get("VOICE_PRELOAD", "1").strip().lower() in (
    "1", "true", "yes", "y", "on",
)
VOICE_PRELOAD_WARM_TTS = os.environ.get("VOICE_PRELOAD_WARM_TTS", "1").strip().lower() in (
    "1", "true", "yes", "y", "on",
)
# Per-turn timing: STT ms, prompt size, LLM time-to-first-token (set 0 to disable)
VOICE_TIMING = os.environ.get("VOICE_TIMING", "1").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

# Chunking — how aggressively to split the LLM stream for early TTS playback
# VOICE_SPLIT_ON_COMMA=1 also splits on commas (shorter first chunk, more TTS calls).
VOICE_SPLIT_ON_COMMA = os.environ.get("VOICE_SPLIT_ON_COMMA", "0").strip().lower() in (
    "1", "true", "yes", "y", "on",
)
SPLIT_PATTERN = re.compile(
    r'(?<=[.!?;,])\s+' if VOICE_SPLIT_ON_COMMA else r'(?<=[.!?;])\s+'
)
# Persist memory on a background thread so the next listen starts sooner
VOICE_ASYNC_MEMORY_SAVE = os.environ.get("VOICE_ASYNC_MEMORY_SAVE", "1").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

# LLM streaming timeouts (seconds) — free models often need >30s for first token
LLM_FIRST_TOKEN_TIMEOUT = max(
    15, int(os.environ.get("VOICE_LLM_FIRST_TOKEN_TIMEOUT", "60")),
)
LLM_INTER_TOKEN_TIMEOUT = max(
    5, int(os.environ.get("VOICE_LLM_INTER_TOKEN_TIMEOUT", "20")),
)
LLM_RETRY_ATTEMPTS      = max(1, int(os.environ.get("VOICE_LLM_RETRY_ATTEMPTS", "3")))
LLM_RETRY_BASE_SECS     = float(os.environ.get("VOICE_LLM_RETRY_BASE_SECS", "2"))

# Conversation context & memory (env-tunable)
MAX_HISTORY_PAIRS       = max(1, int(os.environ.get("VOICE_MAX_HISTORY_PAIRS", "12")))
MAX_CONTEXT_CHARS       = int(os.environ.get("VOICE_MAX_CONTEXT_CHARS", "10000"))  # 0 = no char cap
MAX_MEMORY_NOTES_CHARS  = int(os.environ.get("VOICE_MEMORY_NOTES_MAX_CHARS", "4000"))
# Total chars sent to OpenRouter (system + turns); 0 = no cap beyond pair/notes limits
MAX_PROMPT_CHARS        = int(os.environ.get("VOICE_MAX_PROMPT_CHARS", "5000"))
MEMORY_FILE_PATH        = os.environ.get("VOICE_MEMORY_PATH", "").strip()
LLM_MAX_TOKENS          = max(64, int(os.environ.get("VOICE_LLM_MAX_TOKENS", "512")))
MEMORY_FILE_VERSION     = 1
# ───────────────────────────────────────────────────────────────────────────────


# ── Audio devices & defaults ───────────────────────────────────────────────────

def _default_in_out() -> Tuple[Optional[int], Optional[int]]:
    """Return (input_device_id, output_device_id) for sounddevice.default.device."""
    d = sd.default.device
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return d[0], d[1]
    if isinstance(d, int):
        return d, d
    return None, None


def resolve_input_device_id(spec: str) -> int:
    """Map VOICE_INPUT_DEVICE string to a PortAudio input device index."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty VOICE_INPUT_DEVICE")
    if spec.isdigit():
        idx = int(spec)
        dev = sd.query_devices(idx)
        if int(dev.get("max_input_channels", 0)) <= 0:
            raise ValueError(f"Device {idx} ({dev.get('name')!r}) is not an input device")
        return idx
    needle = spec.lower()
    matches: List[Tuple[int, str]] = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        name = str(dev.get("name", ""))
        if needle in name.lower():
            matches.append((i, name))
    if not matches:
        raise ValueError(
            f"No input device name contains {spec!r}. "
            "Set VOICE_LIST_DEVICES=1 and pick an index with VOICE_INPUT_DEVICE=<n>."
        )
    if len(matches) > 1:
        lines = "\n".join(f"  [{i}] {n}" for i, n in matches[:15])
        more = "" if len(matches) <= 15 else f"\n  … and {len(matches) - 15} more"
        raise ValueError(
            f"Ambiguous VOICE_INPUT_DEVICE={spec!r}; multiple matches:\n{lines}{more}\n"
            "Use a longer substring or an integer index."
        )
    return matches[0][0]


def apply_input_device(device_index: Optional[int]) -> None:
    """Set default input device; leave output unchanged."""
    if device_index is None:
        return
    _in, out = _default_in_out()
    sd.default.device = (device_index, out)


def print_input_device_list() -> None:
    """Print all input-capable devices (for VOICE_LIST_DEVICES=1)."""
    print("\n--- Input-capable audio devices (use VOICE_INPUT_DEVICE=<index or substring>) ---", flush=True)
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        ch = dev.get("max_input_channels")
        sr = dev.get("default_samplerate")
        print(f"  [{i}] {dev.get('name')}  (in_ch={ch}, default_sr={sr})", flush=True)
    print("--- end device list ---\n", flush=True)


def describe_default_input() -> str:
    """One-line summary of the current default input device."""
    idx, _ = _default_in_out()
    if idx is None:
        return "default input (system)"
    try:
        dev = sd.query_devices(idx)
        return f"[{idx}] {dev.get('name', '?')}"
    except Exception:
        return f"[{idx}]"


# ── Audio recording ────────────────────────────────────────────────────────────

def record_until_silence() -> Tuple[Optional[np.ndarray], str]:
    """Stream mic input; return (audio_or_None, status_message for logging)."""
    print("\n[Listening ...]", flush=True)

    audio_chunks: List[np.ndarray] = []
    speech_chunks = 0
    silent_chunks = 0
    has_speech    = False
    stop_event    = threading.Event()

    silence_limit = int(SILENCE_SECS * SAMPLE_RATE / BLOCK_SIZE)
    min_speech    = max(1, int(MIN_SPEECH_SECS * SAMPLE_RATE / BLOCK_SIZE))

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, speech_chunks, has_speech
        if status:
            pass  # optional: could log underflow
        chunk = indata[:, 0].copy()
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > SILENCE_THRESHOLD:
            has_speech    = True
            speech_chunks += 1
            silent_chunks  = 0
        elif has_speech:
            silent_chunks += 1
        if has_speech:
            audio_chunks.append(chunk)
        if has_speech and silent_chunks >= silence_limit:
            if speech_chunks >= min_speech:
                stop_event.set()
            else:
                # blip too short (e.g. speaker echo) — reset and keep listening
                audio_chunks.clear()
                speech_chunks = 0
                silent_chunks = 0
                has_speech    = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        blocksize=BLOCK_SIZE, dtype="float32", callback=callback,
    ):
        stopped_by_event = stop_event.wait(timeout=MAX_RECORD_SECS)

    if not has_speech:
        if stopped_by_event:
            return None, (
                "No speech captured after internal stop (unexpected). "
                f"Try lowering VOICE_SILENCE_THRESHOLD (now {SILENCE_THRESHOLD})."
            )
        return None, (
            f"No input above threshold for {MAX_RECORD_SECS:g}s "
            f"(VOICE_SILENCE_THRESHOLD={SILENCE_THRESHOLD}). "
            "Check mic permission, VOICE_INPUT_DEVICE, or lower the threshold."
        )
    if speech_chunks < min_speech:
        return None, (
            f"Speech too short ({speech_chunks} loud blocks < {min_speech} required). "
            f"Speak longer or lower VOICE_MIN_SPEECH_SECS (now {MIN_SPEECH_SECS}s)."
        )
    if not audio_chunks:
        return None, "No audio buffers captured (unexpected); try again."
    return np.concatenate(audio_chunks), "ok"


def wait_after_playback() -> None:
    """Brief pause after TTS so room/speaker echo does not trigger VAD."""
    if POST_PLAYBACK_SLEEP > 0:
        time.sleep(POST_PLAYBACK_SLEEP)


# ── STT ────────────────────────────────────────────────────────────────────────


def _parse_stt_argv() -> Optional[str]:
    """Read --stt / --asr from argv (does not mutate argv)."""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a in ("--stt", "--asr") and i + 1 < len(args):
            return args[i + 1].strip()
        if a.startswith("--stt=") or a.startswith("--asr="):
            _, _, rest = a.partition("=")
            return rest.strip()
    return None


def _normalize_stt_backend(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s in ("qwen", "qwen3", "mlx", "qwen-asr", "qwen_asr"):
        return "qwen"
    if s in ("whisper", "faster-whisper", "faster_whisper", "fw", "openai-whisper"):
        return "whisper"
    return None


def resolve_stt_backend() -> str:
    """
    Pick STT engine: CLI --stt/--asr, then VOICE_STT_BACKEND, then interactive if TTY, else qwen.
    """
    cli = _parse_stt_argv()
    if cli is not None:
        b = _normalize_stt_backend(cli)
        if b is None:
            sys.exit(f"Invalid --stt/--asr={cli!r}. Use qwen or whisper.")
        return b

    env_raw = os.environ.get("VOICE_STT_BACKEND", "").strip()
    if env_raw:
        b = _normalize_stt_backend(env_raw)
        if b is None:
            sys.exit(f"Invalid VOICE_STT_BACKEND={env_raw!r}. Use qwen or whisper.")
        return b

    if sys.stdin.isatty() and sys.stdout.isatty():
        print(
            "\nSelect speech-to-text engine:\n"
            "  1) Qwen3-ASR (MLX on Apple Silicon, default)\n"
            "  2) Faster-Whisper (CPU/GPU via CTranslate2)\n",
            end="",
            flush=True,
        )
        choice = input("Enter 1 or 2 [1]: ").strip() or "1"
        if choice in ("2", "w", "W", "whisper", "Whisper"):
            return "whisper"
        return "qwen"

    return "qwen"


def transcribe_qwen(model: Any, audio: np.ndarray) -> str:
    result = model.transcribe((audio, SAMPLE_RATE), language="en")
    text = (getattr(result, "text", None) or "").strip()
    text = text.replace("\ufeff", "")
    text = text.translate(str.maketrans("", "", "\u200b\u200c\u200d\u2060"))
    return text.strip()


def transcribe_whisper(model: Any, audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    segments, _info = model.transcribe(
        arr,
        language="en",
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=False,
    )
    text = "".join(seg.text for seg in segments).strip()
    text = text.replace("\ufeff", "")
    text = text.translate(str.maketrans("", "", "\u200b\u200c\u200d\u2060"))
    return text.strip()


def load_stt_transcriber(backend: str) -> Tuple[str, Callable[[np.ndarray], str]]:
    """Load the chosen STT model; return (label, transcribe_fn)."""
    if backend == "qwen":
        from mlx_qwen3_asr.session import Session as ASRSession

        print(f"Loading Qwen3-ASR ({QWEN_MODEL}) ...", flush=True)
        qwen = ASRSession(QWEN_MODEL)
        label = f"Qwen3-ASR ({QWEN_MODEL})"
        return label, lambda a: transcribe_qwen(qwen, a)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        sys.exit(
            "faster-whisper is not installed. Install with:\n"
            "  pip install faster-whisper\n"
            f" ({exc})"
        )

    dev = WHISPER_DEVICE
    if dev.lower() in ("auto", "default", ""):
        dev = "auto"
    ctype = WHISPER_COMPUTE_TYPE
    if ctype.lower() in ("auto", "default", ""):
        ctype = "default"

    print(
        f"Loading Faster-Whisper (model={WHISPER_MODEL!r}, device={dev!r}, compute_type={ctype!r}) ...",
        flush=True,
    )
    whisper = WhisperModel(WHISPER_MODEL, device=dev, compute_type=ctype)
    label = f"Faster-Whisper ({WHISPER_MODEL})"
    return label, lambda a: transcribe_whisper(whisper, a)


# ── LLM (streaming) ────────────────────────────────────────────────────────────


@dataclass
class LLMProviderConfig:
    """OpenAI-style chat/completions provider (OpenRouter or Cerebras)."""
    provider: str
    api_key: str
    model: str
    chat_url: str
    extra_headers: Dict[str, str] = field(default_factory=dict)
    session: requests.Session = field(default_factory=requests.Session)

    def label(self) -> str:
        return f"{self.provider} ({self.model})"


def _parse_llm_argv() -> Optional[str]:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--llm" and i + 1 < len(args):
            return args[i + 1].strip()
        if a.startswith("--llm="):
            _, _, rest = a.partition("=")
            return rest.strip()
    return None


def _normalize_llm_backend(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s in ("openrouter", "or", "router"):
        return "openrouter"
    if s in ("cerebras", "cerebras-ai", "cerebras_ai", "cb"):
        return "cerebras"
    return None


def resolve_llm_config() -> LLMProviderConfig:
    """CLI --llm, then VOICE_LLM_BACKEND, then interactive menu if TTY, else openrouter."""
    cli = _parse_llm_argv()
    if cli is not None:
        backend = _normalize_llm_backend(cli)
        if backend is None:
            sys.exit(f"Invalid --llm={cli!r}. Use openrouter or cerebras.")
    else:
        env_raw = os.environ.get("VOICE_LLM_BACKEND", "").strip()
        if env_raw:
            backend = _normalize_llm_backend(env_raw)
            if backend is None:
                sys.exit(f"Invalid VOICE_LLM_BACKEND={env_raw!r}. Use openrouter or cerebras.")
        elif sys.stdin.isatty() and sys.stdout.isatty():
            print(
                "\nSelect LLM provider:\n"
                "  1) OpenRouter (default)\n"
                "  2) Cerebras Inference (fast hosted API)\n",
                end="",
                flush=True,
            )
            choice = input("Enter 1 or 2 [1]: ").strip() or "1"
            backend = "cerebras" if choice in ("2", "c", "C", "cerebras") else "openrouter"
        else:
            backend = "openrouter"

    if backend == "cerebras":
        if not CEREBRAS_API_KEY:
            sys.exit(
                "Error: CEREBRAS_API_KEY is not set.\n"
                "  export CEREBRAS_API_KEY='csk-...'   # https://cloud.cerebras.ai"
            )
        return LLMProviderConfig(
            provider="cerebras",
            api_key=CEREBRAS_API_KEY,
            model=CEREBRAS_MODEL or "llama3.1-8b",
            chat_url=f"{CEREBRAS_BASE_URL}/chat/completions",
        )

    if not OPENROUTER_API_KEY:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'   # https://openrouter.ai"
        )
    return LLMProviderConfig(
        provider="openrouter",
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        chat_url="https://openrouter.ai/api/v1/chat/completions",
        extra_headers={
            "HTTP-Referer": "https://voice-agent.local",
            "X-Title": "Voice Agent",
        },
    )


def _openrouter_reasoning_body() -> Optional[Dict[str, Any]]:
    """
    Build OpenRouter `reasoning` object when the model supports it.
    When disabled (default), return None and omit the field — many free models 400 on
    reasoning.effort=none.
    """
    raw = OPENROUTER_REASONING_EFFORT
    if raw in ("off", "false", "0", "no", "disable", "disabled", "none", ""):
        return None
    if raw in ("exclude", "hidden"):
        return {"exclude": True}
    if raw in ("xhigh", "high", "medium", "low", "minimal"):
        return {"effort": raw}
    if raw.isdigit():
        return {"max_tokens": int(raw)}
    if raw in ("on", "true", "1", "default", "medium"):
        return {"enabled": True}
    return {"effort": raw}


def _llm_api_error_detail(resp: requests.Response) -> str:
    """Short human-readable message from an OpenAI-style error response."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code")
                if msg:
                    return str(msg)
            if data.get("message"):
                return str(data["message"])
    except (json.JSONDecodeError, ValueError):
        pass
    text = (resp.text or "").strip()
    if text and len(text) < 500:
        return text
    return resp.reason or f"HTTP {resp.status_code}"


def _cerebras_request_options(model: str) -> Dict[str, Any]:
    """Extra Cerebras fields for reasoning models (ignored by plain llama models)."""
    m = model.lower()
    opts: Dict[str, Any] = {}
    if "gpt-oss" in m:
        if CEREBRAS_REASONING_FORMAT:
            opts["reasoning_format"] = CEREBRAS_REASONING_FORMAT
        if CEREBRAS_REASONING_EFFORT:
            opts["reasoning_effort"] = CEREBRAS_REASONING_EFFORT
    elif "glm" in m:
        if CEREBRAS_REASONING_FORMAT:
            opts["reasoning_format"] = CEREBRAS_REASONING_FORMAT
        if CEREBRAS_REASONING_EFFORT:
            opts["reasoning_effort"] = CEREBRAS_REASONING_EFFORT
    return opts


def _stream_delta_speakable_text(delta_obj: Dict[str, Any]) -> str:
    """Text to speak from one streaming delta (content only — never reasoning)."""
    if not delta_obj:
        return ""
    content = delta_obj.get("content")
    if isinstance(content, str) and content:
        return content
    return ""


def sanitize_text_for_speech(text: str) -> str:
    """Strip ASCII-art / empty lines so TTS does not choke on board grids."""
    if not text or not str(text).strip():
        return ""
    lines_out: List[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # e.g. "1 | 2 | 3" or "| X |"
        if "|" in stripped and re.fullmatch(r"[\s|\dXxOo\-]+", stripped):
            continue
        lines_out.append(stripped)
    if lines_out:
        return " ".join(lines_out)
    return " ".join(str(text).split())


def stream_llm(
    messages: List[Dict],
    llm: LLMProviderConfig,
    *,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Iterator[str]:
    """Yield text tokens from a streaming chat completion (OpenRouter or Cerebras).

    Runs the HTTP fetch in a daemon thread so requests' per-read socket
    timeout cannot cause an uninterruptible hang in the caller.
    """
    token_q: "queue.Queue[object]" = queue.Queue()
    model = llm.model
    provider = llm.provider

    def _fetch() -> None:
        try:
            headers = {
                "Authorization": f"Bearer {llm.api_key}",
                "Content-Type": "application/json",
                **llm.extra_headers,
            }
            body: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            }
            reasoning: Optional[Dict[str, Any]] = None
            if provider == "openrouter":
                reasoning = _openrouter_reasoning_body()
                if reasoning is not None:
                    body["reasoning"] = reasoning
            elif provider == "cerebras":
                body.update(_cerebras_request_options(model))
            got_speakable = False
            resp = None
            for attempt in range(LLM_RETRY_ATTEMPTS):
                resp = llm.session.post(
                    llm.chat_url, headers=headers, json=body, stream=True, timeout=30,
                )
                if resp.status_code == 429 and attempt + 1 < LLM_RETRY_ATTEMPTS:
                    wait = LLM_RETRY_BASE_SECS * (2 ** attempt)
                    print(
                        f"[LLM]: rate limited (429) on {provider} model={model!r}; "
                        f"retry {attempt + 2}/{LLM_RETRY_ATTEMPTS} in {wait:.0f}s ...",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                break
            assert resp is not None
            if resp.status_code == 429:
                if provider == "openrouter":
                    print(
                        "[LLM]: OpenRouter returned 429 Too Many Requests. "
                        "Free models are often rate-limited — wait, change OPENROUTER_MODEL, "
                        "or add credits at openrouter.ai.",
                        flush=True,
                    )
                else:
                    print(
                        "[LLM]: Cerebras returned 429 Too Many Requests. "
                        "Wait and retry, or check usage at cloud.cerebras.ai.",
                        flush=True,
                    )
            elif resp.status_code == 400:
                detail = _llm_api_error_detail(resp)
                print(
                    f"[LLM]: {provider} 400 Bad Request for model={model!r}: {detail}",
                    flush=True,
                )
                if provider == "openrouter" and reasoning is not None:
                    print(
                        "[LLM]: Tip: set OPENROUTER_REASONING_EFFORT=none to omit the reasoning field.",
                        flush=True,
                    )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data.strip() == b"[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta_obj = obj["choices"][0].get("delta") or {}
                        delta = _stream_delta_speakable_text(delta_obj)
                        if delta:
                            got_speakable = True
                            token_q.put(delta)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            if not got_speakable:
                hint = ""
                if provider == "cerebras" and "gpt-oss" in model.lower():
                    hint = (
                        " (gpt-oss streams thinking in delta.reasoning by default; "
                        "use CEREBRAS_REASONING_FORMAT=hidden or CEREBRAS_MODEL=llama3.1-8b)"
                    )
                token_q.put(
                    RuntimeError(f"LLM returned no speakable content{hint}")
                )
        except Exception as e:
            token_q.put(e)
        finally:
            token_q.put(None)

    threading.Thread(target=_fetch, daemon=True).start()

    first = True
    while True:
        wait = LLM_FIRST_TOKEN_TIMEOUT if first else LLM_INTER_TOKEN_TIMEOUT
        try:
            item = token_q.get(timeout=wait)
        except queue.Empty:
            label = "first token" if first else "next token"
            raise TimeoutError(f"LLM timed out waiting for {label} ({wait}s)")
        first = False
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


# ── TTS chunking pipeline ──────────────────────────────────────────────────────

def iter_sentences(token_stream: Iterator[str]) -> Iterator[str]:
    """
    Buffer streaming tokens into sentence-level chunks.
    Yields a chunk as soon as sentence-ending punctuation + whitespace appears,
    so Kokoro can start synthesising before the LLM has finished.
    """
    buf = ""
    for token in token_stream:
        buf += token
        while True:
            m = SPLIT_PATTERN.search(buf)
            if not m:
                break
            chunk = buf[: m.start() + 1].strip()
            buf   = buf[m.end():]
            if chunk:
                yield chunk
    if buf.strip():
        yield buf.strip()


def _to_numpy(audio) -> np.ndarray:
    return audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)


def ensure_spacy_english_model() -> None:
    """Kokoro/misaki needs en_core_web_sm; download once if missing (can take a minute)."""
    try:
        import spacy
    except ImportError as exc:
        sys.exit(f"spaCy is required for Kokoro TTS but is not installed ({exc}).")

    name = "en_core_web_sm"
    if spacy.util.is_package(name):
        return
    print(
        f"[Setup]: downloading spaCy model {name!r} (required for Kokoro; first time only) ...",
        flush=True,
    )
    try:
        spacy.cli.download(name)
    except Exception as exc:
        sys.exit(
            f"Could not install spaCy model {name!r}: {exc}\n"
            f"  Run: {sys.executable} -m spacy download {name}"
        )


def load_kokoro_pipeline() -> "KPipeline":
    """Load Kokoro TTS (imports torch/spacy on first call — may take 30–90s)."""
    ensure_spacy_english_model()
    print(
        "Loading Kokoro TTS (torch + spaCy; first run can take up to ~2 min) ...",
        flush=True,
    )
    from kokoro import KPipeline

    return KPipeline(lang_code=KOKORO_LANG, repo_id=KOKORO_REPO_ID)


def _kokoro_synthesize(pipeline: "KPipeline", text: str) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        if audio is not None:
            chunks.append(_to_numpy(audio))
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def openai_speech_synthesize(text: str) -> Tuple[np.ndarray, int]:
    """
    Call OpenAI-compatible POST {base_url}/audio/speech.
    Returns (float32 mono audio, sample_rate).
    """
    if not text.strip():
        return np.array([], dtype=np.float32), OPENAI_SPEECH_SAMPLE_RATE
    if not OPENAI_SPEECH_API_KEY:
        raise ValueError(
            "OPENAI_SPEECH_API_KEY or OPENAI_API_KEY is required for OpenAI TTS "
            "(set VOICE_TTS_BACKEND=openai)."
        )

    url = f"{OPENAI_SPEECH_BASE_URL}/audio/speech"
    payload: Dict[str, Any] = {
        "model": OPENAI_SPEECH_MODEL,
        "input": text,
        "voice": OPENAI_SPEECH_VOICE,
        "response_format": OPENAI_SPEECH_FORMAT,
    }
    if OPENAI_SPEECH_SPEED != 1.0:
        payload["speed"] = OPENAI_SPEECH_SPEED

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENAI_SPEECH_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()

    if OPENAI_SPEECH_FORMAT == "pcm":
        pcm = np.frombuffer(resp.content, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        return audio, OPENAI_SPEECH_SAMPLE_RATE

    import io
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.ndim > 1 else data.reshape(-1)
    return mono.astype(np.float32), int(sr)


def _wav_bytes_to_mono_float(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.size else data.reshape(-1)
    return mono.astype(np.float32), int(sr)


def _json_list_items(payload: Any, id_keys: Tuple[str, ...] = ("id", "name", "code")) -> List[str]:
    """Extract string ids from assorted list/dict API response shapes."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        out: List[str] = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in id_keys:
                    if item.get(key):
                        out.append(str(item[key]))
                        break
        return out
    if isinstance(payload, dict):
        for key in ("data", "items", "models", "languages", "voices", "results"):
            if key in payload:
                return _json_list_items(payload[key], id_keys)
    return []


def _openvox_preload_model_ids(available: List[str], active_model: str) -> List[str]:
    """Which OpenVox server models to POST /models/{id}/load at startup."""
    spec = OPENVox_PRELOAD_MODELS
    if not spec or spec.lower() == "active":
        return [active_model]
    if spec.lower() == "all":
        return list(available)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    out = [m for m in wanted if m in available]
    missing = [m for m in wanted if m not in available]
    if missing:
        print(
            f"[OpenVox]: OPENVox_PRELOAD_MODELS — not on server (skipped): {', '.join(missing)}",
            flush=True,
        )
    return out or [active_model]


class OpenVoxTTS:
    """
    OpenVox local voice API: discover model, warm load, resolve language/voice,
    then POST /audio/speech (WAV or SSE stream).
    """

    def __init__(self) -> None:
        self.base_url = OPENVox_BASE_URL
        self.model = OPENVox_MODEL
        self.language = OPENVox_LANGUAGE
        self.voice = OPENVox_VOICE or None
        self.use_stream = OPENVox_STREAM
        self.sample_rate = 24000
        self._models_loaded: set[str] = set()
        self.label = "OpenVox"

    def setup(self) -> bool:
        """Probe API, pick model/language/voice, warm-load model. Returns False if unavailable."""
        try:
            resp = self._request("GET", "/models", timeout=15)
            resp.raise_for_status()
            models = _json_list_items(resp.json())
            if not models:
                print("[OpenVox]: GET /models returned no models.", flush=True)
                return False
            if not self.model:
                self.model = models[0]
                if len(models) > 1:
                    print(f"[OpenVox]: models available: {', '.join(models)}", flush=True)
            elif self.model not in models:
                print(
                    f"[OpenVox]: OPENVox_MODEL={self.model!r} not in {models}; using {models[0]!r}.",
                    flush=True,
                )
                self.model = models[0]

            preload_ids = _openvox_preload_model_ids(models, self.model)
            print(
                f"[OpenVox]: preloading {len(preload_ids)} model(s): {', '.join(preload_ids)}",
                flush=True,
            )
            for model_id in preload_ids:
                self._warm_load_model(model_id)
            self._resolve_language_and_voice()
            mode = "SSE stream" if self.use_stream else "WAV"
            self.label = f"OpenVox ({self.model}/{self.language}/{self.voice}, {mode})"
            print(f"[OpenVox]: ready — {self.label}", flush=True)
            return True
        except requests.RequestException as exc:
            print(
                f"[OpenVox]: cannot reach voice API at {self.base_url!r} ({exc}).",
                flush=True,
            )
            return False

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = self._url(path)
        timeout = kwargs.pop("timeout", 120)
        last: Optional[requests.Response] = None
        for attempt in range(OPENVox_MAX_RETRIES):
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            last = resp
            if resp.status_code != 429:
                return resp
            wait = OPENVox_429_WAIT * (attempt + 1)
            print(
                f"[OpenVox]: server busy (429); waiting {wait:.0f}s before retry "
                f"({attempt + 1}/{OPENVox_MAX_RETRIES}) ...",
                flush=True,
            )
            time.sleep(wait)
        assert last is not None
        return last

    def _warm_load_model(self, model_id: str) -> None:
        if model_id in self._models_loaded:
            return
        print(f"[OpenVox]: loading model {model_id!r} ...", flush=True)
        resp = self._request("POST", f"/models/{model_id}/load", timeout=300)
        resp.raise_for_status()
        self._models_loaded.add(model_id)

    def _resolve_language_and_voice(self) -> None:
        assert self.model
        resp = self._request("GET", f"/models/{self.model}/languages", timeout=30)
        resp.raise_for_status()
        languages = _json_list_items(resp.json())
        if not languages:
            languages = [self.language]
        if self.language not in languages:
            print(
                f"[OpenVox]: language {self.language!r} not in {languages}; using {languages[0]!r}.",
                flush=True,
            )
            self.language = languages[0]

        self._refresh_voice(preferred=self.voice)

    def _refresh_voice(self, preferred: Optional[str] = None) -> None:
        assert self.model
        resp = self._request(
            "GET",
            f"/models/{self.model}/voices",
            params={"language": self.language},
            timeout=30,
        )
        resp.raise_for_status()
        voices = _json_list_items(resp.json())
        if not voices:
            raise ValueError(
                f"No voices for model={self.model!r} language={self.language!r}"
            )
        pick = preferred if preferred and preferred in voices else voices[0]
        if preferred and preferred not in voices:
            print(
                f"[OpenVox]: voice {preferred!r} not available; using {pick!r} "
                f"(choices: {', '.join(voices[:8])}{'…' if len(voices) > 8 else ''}).",
                flush=True,
            )
        self.voice = pick

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.array([], dtype=np.float32)
        assert self.model and self.voice
        self._warm_load_model(self.model)
        try:
            if self.use_stream:
                return self._synthesize_stream(text)
            return self._synthesize_wav(text)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (400, 404, 422):
                self._refresh_voice(preferred=None)
                if self.use_stream:
                    return self._synthesize_stream(text)
                return self._synthesize_wav(text)
            raise

    def _speech_body(self, text: str, *, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "input": text,
            "language": self.language,
            "voice": self.voice,
        }
        if stream:
            body["stream"] = True
        else:
            body["response_format"] = "wav"
        return body

    def _synthesize_wav(self, text: str) -> np.ndarray:
        resp = self._request(
            "POST",
            "/audio/speech",
            headers={"Content-Type": "application/json"},
            json=self._speech_body(text, stream=False),
            timeout=300,
        )
        resp.raise_for_status()
        audio, sr = _wav_bytes_to_mono_float(resp.content)
        self.sample_rate = sr
        return audio

    def _synthesize_stream(self, text: str) -> np.ndarray:
        resp = self._request(
            "POST",
            "/audio/speech",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            json=self._speech_body(text, stream=True),
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        parts: List[np.ndarray] = []
        event_name: Optional[str] = None
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            ev = event_name or obj.get("type") or obj.get("event")
            if ev in ("audio.chunk", "response.audio.chunk"):
                chunk_obj = obj.get("data") if isinstance(obj.get("data"), dict) else obj
                b64 = None
                if isinstance(chunk_obj, dict):
                    b64 = chunk_obj.get("audio") or chunk_obj.get("data")
                if not b64:
                    continue
                wav_bytes = base64.b64decode(b64)
                audio, sr = _wav_bytes_to_mono_float(wav_bytes)
                self.sample_rate = sr
                parts.append(audio)
            event_name = None
        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)


def _playback_sample_rate(sample_rate: SampleRateSpec) -> int:
    return sample_rate() if callable(sample_rate) else sample_rate


def _speak_line(
    text: str,
    synthesize: Callable[[str], np.ndarray],
    sample_rate: SampleRateSpec,
) -> None:
    print(f"[Assistant]: {text}", flush=True)
    audio = synthesize(text)
    if audio.size:
        sd.play(audio, samplerate=_playback_sample_rate(sample_rate), blocking=True)


def speak_chunked(
    sentence_iter: Iterator[str],
    synthesize: Callable[[str], np.ndarray],
    sample_rate: SampleRateSpec,
) -> str:
    """
    Pipeline TTS synthesis with playback so they overlap:
      synth thread processes sentence N+1 while main thread plays sentence N.

    Returns the full assistant reply text for conversation history.
    """
    audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=3)
    reply_parts: List[str] = []
    synth_error: List[Optional[BaseException]] = [None]

    header_printed = False

    def synth_worker() -> None:
        nonlocal header_printed
        try:
            for sentence in sentence_iter:
                if not header_printed:
                    print("[Assistant]: ", end="", flush=True)
                    header_printed = True
                spoken = sanitize_text_for_speech(sentence)
                if not spoken:
                    continue
                reply_parts.append(spoken)
                print(spoken, end=" ", flush=True)
                audio = synthesize(spoken)
                if audio.size:
                    audio_q.put(audio)
        except Exception as e:
            synth_error[0] = e
            # LLM timeouts surface here via the sentence iterator; main prints once.
            if not (isinstance(e, TimeoutError) and "LLM timed out" in str(e)):
                print(f"\n[TTS / stream error]: {e}", flush=True)
        finally:
            audio_q.put(None)

    synth_thread = threading.Thread(target=synth_worker, daemon=True)
    synth_thread.start()

    while True:
        try:
            chunk_audio = audio_q.get(timeout=60)
        except queue.Empty:
            print("\n[TTS]: timed out waiting for audio (queue empty).", flush=True)
            synth_thread.join(timeout=15)
            if synth_error[0] is None:
                synth_error[0] = TimeoutError("TTS playback queue timed out (no audio chunk in 60s)")
            break
        if chunk_audio is None:
            break
        sd.play(chunk_audio, samplerate=_playback_sample_rate(sample_rate), blocking=True)

    synth_thread.join(timeout=30)
    err = synth_error[0]
    print(flush=True)
    if err is not None:
        raise err
    return " ".join(reply_parts)


@dataclass
class TTSEngine:
    """Text-to-speech backend (Kokoro, OpenAI Speech API, or OpenVox)."""
    label: str
    sample_rate: int
    speak: Callable[[str], None]
    speak_chunked: Callable[[Iterator[str]], str]
    warm_up: Optional[Callable[[], None]] = None


def run_tts_warm_up(tts: TTSEngine) -> None:
    """One silent synthesis pass so the first real reply is not cold-start slow."""
    if not VOICE_PRELOAD_WARM_TTS or tts.warm_up is None:
        return
    print("[Preload]: warming TTS pipeline ...", flush=True)
    try:
        tts.warm_up()
        print("[Preload]: TTS warm-up done.", flush=True)
    except Exception as exc:
        print(f"[Preload]: TTS warm-up failed ({exc}); continuing.", flush=True)


def load_text_only_tts(reason: str) -> TTSEngine:
    """Fallback when local/cloud TTS is unavailable — text replies only."""
    _warned = False

    def _maybe_warn() -> None:
        nonlocal _warned
        if not _warned:
            print(f"[TTS]: {reason}", flush=True)
            _warned = True

    def speak(text: str) -> None:
        print(f"[Assistant]: {text}", flush=True)
        _maybe_warn()

    def speak_chunked(sentence_iter: Iterator[str]) -> str:
        parts: List[str] = []
        for sentence in sentence_iter:
            if not parts:
                print("[Assistant]: ", end="", flush=True)
            parts.append(sentence)
            print(sentence, end=" ", flush=True)
        if parts:
            print(flush=True)
        _maybe_warn()
        return " ".join(parts)

    return TTSEngine(
        label="text-only (no TTS)",
        sample_rate=24000,
        speak=speak,
        speak_chunked=speak_chunked,
    )


def _parse_tts_argv() -> Optional[str]:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a in ("--tts",) and i + 1 < len(args):
            return args[i + 1].strip()
        if a.startswith("--tts="):
            _, _, rest = a.partition("=")
            return rest.strip()
    return None


def _normalize_tts_backend(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s in ("kokoro", "local"):
        return "kokoro"
    if s in ("openai", "speech", "openai-speech", "openai_speech", "api"):
        return "openai"
    if s in ("openvox", "vox", "local-api", "local_api"):
        return "openvox"
    return None


def resolve_tts_backend() -> str:
    """CLI --tts, then VOICE_TTS_BACKEND, then interactive menu if TTY, else kokoro."""
    cli = _parse_tts_argv()
    if cli is not None:
        b = _normalize_tts_backend(cli)
        if b is None:
            sys.exit(f"Invalid --tts={cli!r}. Use kokoro, openai, or openvox.")
        return b

    env_raw = os.environ.get("VOICE_TTS_BACKEND", "").strip()
    if env_raw:
        b = _normalize_tts_backend(env_raw)
        if b is None:
            sys.exit(f"Invalid VOICE_TTS_BACKEND={env_raw!r}. Use kokoro, openai, or openvox.")
        return b

    if sys.stdin.isatty() and sys.stdout.isatty():
        print(
            "\nSelect text-to-speech engine:\n"
            "  1) Kokoro (local MLX, default)\n"
            "  2) OpenAI Speech API (cloud)\n"
            "  3) OpenVox (local API at 127.0.0.1:8000)\n",
            end="",
            flush=True,
        )
        choice = input("Enter 1, 2, or 3 [1]: ").strip() or "1"
        if choice in ("3", "v", "V", "openvox", "vox"):
            return "openvox"
        if choice in ("2", "o", "O", "openai", "speech"):
            return "openai"
        return "kokoro"

    return "kokoro"


def load_tts_engine(backend: str) -> TTSEngine:
    if backend == "openvox":
        print(f"Connecting to OpenVox at {OPENVox_BASE_URL} ...", flush=True)
        client = OpenVoxTTS()
        if not client.setup():
            return load_text_only_tts(
                "Local OpenVox voice output is unavailable; continuing in text-only mode."
            )
        sr = client.sample_rate

        def _synth(text: str) -> np.ndarray:
            return client.synthesize(text)

        def _warm() -> None:
            client.synthesize("Ready.")

        return TTSEngine(
            label=client.label,
            sample_rate=sr,
            speak=lambda t: _speak_line(t, _synth, lambda: client.sample_rate),
            speak_chunked=lambda it: speak_chunked(it, _synth, lambda: client.sample_rate),
            warm_up=_warm,
        )

    if backend == "openai":
        print(
            f"Using OpenAI Speech API ({OPENAI_SPEECH_BASE_URL}, "
            f"model={OPENAI_SPEECH_MODEL!r}, voice={OPENAI_SPEECH_VOICE!r}) ...",
            flush=True,
        )
        if not OPENAI_SPEECH_API_KEY:
            sys.exit(
                "Error: OpenAI TTS requires OPENAI_SPEECH_API_KEY or OPENAI_API_KEY.\n"
                "  export OPENAI_API_KEY='sk-...'\n"
                "  # optional custom host:\n"
                "  export OPENAI_SPEECH_BASE_URL='https://api.openai.com/v1'"
            )

        def _synth(text: str) -> np.ndarray:
            audio, _sr = openai_speech_synthesize(text)
            return audio

        sr = OPENAI_SPEECH_SAMPLE_RATE
        label = f"OpenAI Speech ({OPENAI_SPEECH_MODEL}/{OPENAI_SPEECH_VOICE})"
        def _warm() -> None:
            openai_speech_synthesize("Ready.")

        return TTSEngine(
            label=label,
            sample_rate=sr,
            speak=lambda t: _speak_line(t, _synth, sr),
            speak_chunked=lambda it: speak_chunked(it, _synth, sr),
            warm_up=_warm,
        )

    pipeline = load_kokoro_pipeline()
    synth = lambda t: _kokoro_synthesize(pipeline, t)

    def _warm() -> None:
        _kokoro_synthesize(pipeline, "Ready.")

    return TTSEngine(
        label=f"Kokoro ({KOKORO_VOICE})",
        sample_rate=KOKORO_SAMPLE_RATE,
        speak=lambda t: _speak_line(t, synth, KOKORO_SAMPLE_RATE),
        speak_chunked=lambda it: speak_chunked(it, synth, KOKORO_SAMPLE_RATE),
        warm_up=_warm,
    )


# ── Conversation helpers ───────────────────────────────────────────────────────

MEMORY_HEADER = "Earlier in this session (compressed):"


def _rest_message_chars(rest: List[Dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in rest)


def llm_context_char_count(messages: List[Dict]) -> int:
    """Total characters sent to OpenRouter (system + all turns)."""
    return sum(len(str(m.get("content", ""))) for m in messages)


def conversation_needs_trim(conversation: List[Dict]) -> bool:
    """True if conversation exceeds configured pair / char / prompt caps."""
    rest = [m for m in conversation if m.get("role") != "system"]
    if len(rest) > MAX_HISTORY_PAIRS * 2:
        return True
    if MAX_CONTEXT_CHARS > 0 and _rest_message_chars(rest) > MAX_CONTEXT_CHARS:
        return True
    if MAX_PROMPT_CHARS > 0 and llm_context_char_count(conversation) > MAX_PROMPT_CHARS:
        return True
    return False


def _one_line(text: str, limit: int = 200) -> str:
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 3)] + "..."


def _trim_notes_from_start(notes: str, *, max_chars: int) -> str:
    """Drop oldest compressed blocks until `notes` fits `max_chars`."""
    notes = notes.strip()
    if len(notes) <= max_chars:
        return notes
    blocks = [b for b in re.split(r"\n\n+", notes) if b.strip()]
    while blocks and len("\n\n".join(blocks)) > max_chars:
        blocks.pop(0)
    return "\n\n".join(blocks).strip()


def append_compressed_turn(memory_notes: str, user_text: str, asst_text: str) -> str:
    """Append a dropped user/assistant pair to the rolling text buffer (bounded)."""
    block = (
        f"User: {_one_line(user_text)}\n"
        f"Assistant: {_one_line(asst_text)}\n"
    )
    notes = f"{memory_notes.rstrip()}\n\n{block}".strip() if memory_notes.strip() else block
    if len(notes) > MAX_MEMORY_NOTES_CHARS:
        notes = notes[-MAX_MEMORY_NOTES_CHARS:]
    return notes


def build_system_message(base_prompt: str, memory_notes: str) -> Dict[str, str]:
    if memory_notes.strip():
        content = (
            f"{base_prompt.rstrip()}\n\n{MEMORY_HEADER}\n{memory_notes.strip()}"
        )
    else:
        content = base_prompt.rstrip()
    return {"role": "system", "content": content}


def trim_history_with_memory(
    messages: List[Dict],
    memory_notes: str,
    *,
    base_system_prompt: str,
    max_pairs: int = MAX_HISTORY_PAIRS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> Tuple[List[Dict], str]:
    """
    Keep up to `max_pairs` recent user/assistant turns and optionally cap total
    characters in those turns. Oldest removed pairs are appended to `memory_notes`
    (bounded) and re-injected via the system message so the model keeps coarse recall.
    When `max_prompt_chars` > 0, also cap system + turns (trim notes, then drop pairs).
    """
    if not messages:
        return [build_system_message(base_system_prompt, "")], ""

    rest = [m for m in messages if m.get("role") != "system"]
    notes = memory_notes.strip()
    dropped_pairs = 0

    def over_pair_cap() -> bool:
        return len(rest) > max_pairs * 2

    def over_char_cap() -> bool:
        return max_context_chars > 0 and _rest_message_chars(rest) > max_context_chars

    while rest and over_pair_cap() and len(rest) >= 2:
        u = rest.pop(0)
        a = rest.pop(0)
        notes = append_compressed_turn(
            notes,
            str(u.get("content", "")),
            str(a.get("content", "")),
        )
        dropped_pairs += 1

    # Char budget: drop additional oldest pairs (same compression path)
    while rest and over_char_cap() and len(rest) >= 2:
        u = rest.pop(0)
        a = rest.pop(0)
        notes = append_compressed_turn(
            notes,
            str(u.get("content", "")),
            str(a.get("content", "")),
        )
        dropped_pairs += 1

    def rebuild() -> List[Dict]:
        return [build_system_message(base_system_prompt, notes)] + rest

    out = rebuild()

    # Hard cap on everything OpenRouter sees (system notes dominate on long sessions)
    while max_prompt_chars > 0 and llm_context_char_count(out) > max_prompt_chars:
        if len(rest) >= 2:
            u = rest.pop(0)
            a = rest.pop(0)
            notes = append_compressed_turn(
                notes,
                str(u.get("content", "")),
                str(a.get("content", "")),
            )
            dropped_pairs += 1
            out = rebuild()
            continue
        if notes:
            room = max(200, max_prompt_chars - len(base_system_prompt) - _rest_message_chars(rest) - 80)
            trimmed = _trim_notes_from_start(notes, max_chars=room)
            if trimmed == notes:
                break
            notes = trimmed
            out = rebuild()
            continue
        break

    if dropped_pairs:
        print(
            f"[Memory]: compressed {dropped_pairs} older turn pair(s); "
            f"keeping {len(rest) // 2} verbatim turn(s) in context.",
            flush=True,
        )

    return out, notes


def load_memory_file(path: str, base_system_prompt: str) -> Tuple[List[Dict], str]:
    """Load persisted turns + memory notes. Returns (conversation_messages, memory_notes)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Memory]: could not load {path!r} ({exc}); starting fresh.", flush=True)
        return [build_system_message(base_system_prompt, "")], ""

    if not isinstance(data, dict):
        return [build_system_message(base_system_prompt, "")], ""

    notes = str(data.get("memory_notes", "") or "").strip()
    turns = data.get("turns")
    if not isinstance(turns, list):
        turns = []

    cleaned: List[Dict] = []
    for m in turns:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if not isinstance(content, str):
            content = str(content)
        cleaned.append({"role": role, "content": content})

    # Drop orphan last message so we only have complete pairs for history
    if len(cleaned) % 2 == 1:
        cleaned.pop()

    if notes or cleaned:
        print(
            f"[Memory]: loaded {len(cleaned) // 2} turn pair(s) from {path!r}.",
            flush=True,
        )

    return [build_system_message(base_system_prompt, notes)] + cleaned, notes


_memory_save_lock = threading.Lock()


def save_memory_file(path: str, memory_notes: str, conversation: List[Dict]) -> None:
    """Persist memory notes and non-system turns."""
    turns = [m for m in conversation if m.get("role") in ("user", "assistant")]
    payload = {
        "version": MEMORY_FILE_VERSION,
        "memory_notes": memory_notes.strip(),
        "turns": turns,
    }
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[Memory]: could not save {path!r} ({exc}).", flush=True)


def schedule_save_memory_file(
    path: str, memory_notes: str, conversation: List[Dict]
) -> None:
    """Save memory synchronously or on a daemon thread (snapshot avoids races)."""
    if not path:
        return
    notes_snap = memory_notes
    conv_snap = [dict(m) for m in conversation]

    def _write() -> None:
        with _memory_save_lock:
            save_memory_file(path, notes_snap, conv_snap)

    if VOICE_ASYNC_MEMORY_SAVE:
        threading.Thread(target=_write, daemon=True).start()
    else:
        _write()


def pop_pending_user_turn(conversation: List[Dict]) -> None:
    """Remove the last message if it is a user turn left after a failed assistant reply."""
    if conversation and conversation[-1].get("role") == "user":
        conversation.pop()


def _normalize_voice_command_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for phrase matching."""
    t = str(text).strip()
    t = t.replace("\ufeff", "")
    t = t.translate(str.maketrans("", "", "\u200b\u200c\u200d\u2060"))
    t = " ".join(t.lower().split())
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


_CLEAR_MEMORY_PHRASES = tuple(
    _normalize_voice_command_text(p)
    for p in (
        "Clear your memory.",
        "Clear my memory.",
        "Clear memory.",
        "Clear memories.",
        "Erase your memory.",
        "Erase my memory.",
        "Reset your memory.",
        "Reset my memory.",
        "Forget everything.",
        "Start fresh.",
        "Wipe your memory.",
        "Empty your memory.",
        "Delete your memory.",
        "Start over.",
        "New conversation.",
    )
)

# STT variants: words between "clear" and "memory"
_MEMORY_CLEAR_RE = re.compile(
    r"(?:"
    r"\bclear\b.{0,48}\bmemor(?:y|ies)\b"
    r"|\berase\b.{0,48}\bmemor(?:y|ies)\b"
    r"|\bwipe\b.{0,48}\bmemor(?:y|ies)\b"
    r"|\breset\b.{0,48}\bmemor(?:y|ies)\b"
    r"|\bforget\b\s+everything\b"
    r"|\bstart\b\s+fresh\b"
    r"|\bempty\b.{0,24}\bmemor(?:y|ies)\b"
    r")",
    re.IGNORECASE,
)

# Runs first on raw STT text — catches normal wording even if normalize would change tokenization.
_QUICK_MEMORY_CLEAR_RE = re.compile(
    r"\bclear\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\bforget\s+everything\b"
    r"|\bstart\s+fresh\b"
    r"|\bwipe\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\berase\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\breset\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\bempty\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\bdelete\s+(?:your|my)\s+memor(?:y|ies)\b"
    r"|\bnew\s+conversation\b"
    r"|\bstart\s+over\b",
    re.IGNORECASE,
)


def user_wants_memory_cleared(text: str) -> bool:
    """True if the user asked to clear session / persisted memory (voice command)."""
    if not (text and str(text).strip()):
        return False
    raw = str(text).strip()
    if _QUICK_MEMORY_CLEAR_RE.search(raw):
        return True
    t = _normalize_voice_command_text(raw)
    if _QUICK_MEMORY_CLEAR_RE.search(t):
        return True
    if any(p and p in t for p in _CLEAR_MEMORY_PHRASES):
        return True
    if _MEMORY_CLEAR_RE.search(raw):
        return True
    return bool(_MEMORY_CLEAR_RE.search(t))


def wipe_all_agent_memory(base_system_prompt: str, memory_path: str) -> Tuple[List[Dict], str]:
    """
    Reset conversation and compressed notes to a clean system-only state.
    When memory_path is set, overwrites the JSON file with empty memory_notes and turns.
    """
    memory_notes = ""
    conversation: List[Dict] = [build_system_message(base_system_prompt, memory_notes)]
    if memory_path.strip():
        save_memory_file(memory_path, memory_notes, conversation)
        tmp = f"{memory_path}.tmp"
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
    return conversation, memory_notes


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Voice agent starting ...", flush=True)

    llm_config = resolve_llm_config()

    if VOICE_LIST_DEVICES:
        print_input_device_list()

    if VOICE_INPUT_DEVICE_RAW:
        try:
            apply_input_device(resolve_input_device_id(VOICE_INPUT_DEVICE_RAW))
        except ValueError as exc:
            sys.exit(f"Error: {exc}")

    print(f"Using audio input device: {describe_default_input()}", flush=True)

    if VOICE_PRELOAD:
        print("\n=== Preloading models ===", flush=True)

    stt_backend = resolve_stt_backend()
    if VOICE_PRELOAD:
        print(f"[Preload]: STT backend={stt_backend!r} ...", flush=True)
    stt_label, stt_transcribe = load_stt_transcriber(stt_backend)
    print(f"Using STT: {stt_label}", flush=True)

    tts_backend = resolve_tts_backend()
    if VOICE_PRELOAD:
        print(f"[Preload]: TTS backend={tts_backend!r} ...", flush=True)
    tts = load_tts_engine(tts_backend)
    print(f"Using TTS: {tts.label}", flush=True)
    if VOICE_PRELOAD:
        run_tts_warm_up(tts)
        print("=== Preload complete ===\n", flush=True)

    if MEMORY_FILE_PATH:
        conversation, memory_notes = load_memory_file(MEMORY_FILE_PATH, SYSTEM_PROMPT)
        conversation, memory_notes = trim_history_with_memory(
            conversation,
            memory_notes,
            base_system_prompt=SYSTEM_PROMPT,
            max_pairs=MAX_HISTORY_PAIRS,
            max_context_chars=MAX_CONTEXT_CHARS,
        )
    else:
        memory_notes = ""
        conversation = [build_system_message(SYSTEM_PROMPT, memory_notes)]

    print(f"\n=== Voice Agent Ready === ({llm_config.label()})", flush=True)
    if MAX_CONTEXT_CHARS > 0:
        print(
            f"Context: up to {MAX_HISTORY_PAIRS} turn pairs, "
            f"~{MAX_CONTEXT_CHARS} chars of recent dialogue, "
            f"memory notes cap {MAX_MEMORY_NOTES_CHARS} chars.",
            flush=True,
        )
    else:
        print(
            f"Context: up to {MAX_HISTORY_PAIRS} turn pairs "
            f"(no char cap); memory notes cap {MAX_MEMORY_NOTES_CHARS} chars.",
            flush=True,
        )
    if MAX_PROMPT_CHARS > 0:
        print(f"Prompt cap: {MAX_PROMPT_CHARS} chars total (system + turns).", flush=True)
    print(f"LLM first-token timeout: {LLM_FIRST_TOKEN_TIMEOUT}s", flush=True)
    if VOICE_TIMING:
        print(
            f"[Timing]: loaded prompt {llm_context_char_count(conversation)} chars",
            flush=True,
        )
    if MEMORY_FILE_PATH:
        print(f"Memory file: {MEMORY_FILE_PATH!r}", flush=True)
    print(f"LLM max_tokens: {LLM_MAX_TOKENS}", flush=True)
    if llm_config.provider == "openrouter":
        print(f"LLM reasoning: {OPENROUTER_REASONING_EFFORT!r} (OpenRouter)", flush=True)
    elif llm_config.provider == "cerebras" and "gpt-oss" in llm_config.model.lower():
        print(
            f"Cerebras reasoning: format={CEREBRAS_REASONING_FORMAT!r}, "
            f"effort={CEREBRAS_REASONING_EFFORT!r}",
            flush=True,
        )
    print("Press Ctrl+C to quit.\n", flush=True)
    tts.speak("Hello! I'm ready. How can I help you?")
    wait_after_playback()

    while True:
        try:
            # 1. Record
            audio, record_status = record_until_silence()
            if audio is None:
                print(f"[Listen skipped]: {record_status}", flush=True)
                continue

            # 2. STT
            t_stt = time.perf_counter()
            print("[Transcribing ...]", flush=True)
            user_text = stt_transcribe(audio)
            stt_ms = (time.perf_counter() - t_stt) * 1000
            if not user_text:
                print("[No speech detected, listening again]", flush=True)
                continue
            print(f"[You]: {user_text}", flush=True)

            if user_wants_memory_cleared(user_text):
                conversation, memory_notes = wipe_all_agent_memory(
                    SYSTEM_PROMPT, MEMORY_FILE_PATH
                )
                print("[Memory]: fully cleared (conversation, compressed notes, disk).", flush=True)
                tts.speak("Okay, I've cleared my memory. We're starting fresh.")
                wait_after_playback()
                continue

            # 3. Stream LLM → sentence chunks → overlapped synth+play
            conversation.append({"role": "user", "content": user_text})
            conversation, memory_notes = trim_history_with_memory(
                conversation,
                memory_notes,
                base_system_prompt=SYSTEM_PROMPT,
                max_pairs=MAX_HISTORY_PAIRS,
                max_context_chars=MAX_CONTEXT_CHARS,
            )
            if VOICE_TIMING:
                print(
                    f"[Timing]: STT {stt_ms:.0f}ms | "
                    f"prompt {llm_context_char_count(conversation)} chars",
                    flush=True,
                )
            print("[Thinking ...]", flush=True)
            t_llm = time.perf_counter()
            try:
                def _timed_token_stream() -> Iterator[str]:
                    first = True
                    for tok in stream_llm(
                        conversation,
                        llm_config,
                        max_tokens=LLM_MAX_TOKENS,
                    ):
                        if first:
                            if VOICE_TIMING:
                                print(
                                    f"[Timing]: LLM first token "
                                    f"{(time.perf_counter() - t_llm) * 1000:.0f}ms",
                                    flush=True,
                                )
                            first = False
                        yield tok

                sentence_iter  = iter_sentences(_timed_token_stream())
                reply          = tts.speak_chunked(sentence_iter)
                if not (reply and reply.strip()):
                    raise RuntimeError(
                        "No speakable reply (empty LLM stream or TTS skipped all chunks)."
                    )
            except requests.HTTPError as e:
                if e.response is not None:
                    print(
                        f"[LLM HTTP error]: {_llm_api_error_detail(e.response)} "
                        f"({llm_config.label()})",
                        flush=True,
                    )
                else:
                    print(f"[LLM HTTP error]: {e}", flush=True)
                pop_pending_user_turn(conversation)
                continue
            except (requests.RequestException, TimeoutError, OSError) as e:
                print(f"[LLM / network error]: {e}", flush=True)
                pop_pending_user_turn(conversation)
                continue
            except Exception as e:
                print(f"[Reply error]: {e}", flush=True)
                pop_pending_user_turn(conversation)
                continue

            conversation.append({"role": "assistant", "content": reply})
            if conversation_needs_trim(conversation):
                conversation, memory_notes = trim_history_with_memory(
                    conversation,
                    memory_notes,
                    base_system_prompt=SYSTEM_PROMPT,
                    max_pairs=MAX_HISTORY_PAIRS,
                    max_context_chars=MAX_CONTEXT_CHARS,
                )
            if MEMORY_FILE_PATH:
                schedule_save_memory_file(MEMORY_FILE_PATH, memory_notes, conversation)
            if VOICE_TIMING:
                print(
                    f"[Timing]: turn total {(time.perf_counter() - t_stt) * 1000:.0f}ms",
                    flush=True,
                )
            wait_after_playback()

        except KeyboardInterrupt:
            print("\n\nGoodbye!", flush=True)
            if MEMORY_FILE_PATH:
                save_memory_file(MEMORY_FILE_PATH, memory_notes, conversation)
            llm_config.session.close()
            tts.speak("Goodbye!")
            break
        except Exception as e:
            print(f"[Unexpected error]: {e}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
