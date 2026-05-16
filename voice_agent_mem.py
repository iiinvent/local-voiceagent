#!/Users/master/lk/.venv/bin/python
"""
Continuous Voice Agent
  STT: Qwen3-ASR via mlx-qwen3-asr (local, Apple Silicon)
  LLM: OpenRouter API  — streamed
  TTS: Kokoro (local, free)  — chunked pipeline

Latency optimisation
  LLM tokens → sentence buffer → [synth thread] → audio queue → [play thread]
  Synthesis of sentence N+1 overlaps with playback of sentence N.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python3 voice_agent.py

Optional environment (see .env.example):
  VOICE_SILENCE_THRESHOLD, VOICE_SILENCE_SECS, VOICE_MIN_SPEECH_SECS, VOICE_MAX_RECORD_SECS
  VOICE_INPUT_DEVICE (PortAudio index or substring of device name)
  VOICE_LIST_DEVICES=1 — print all input-capable devices at startup
  VOICE_POST_PLAYBACK_SLEEP — seconds to wait after TTS before listening (reduces speaker echo)
  VOICE_MAX_HISTORY_PAIRS, VOICE_MAX_CONTEXT_CHARS, VOICE_MEMORY_NOTES_MAX_CHARS — context window
  VOICE_MEMORY_PATH — JSON file to persist memory + recent turns across runs (empty = off)
  VOICE_LLM_MAX_TOKENS — completion cap for spoken replies
"""

import json
import os
import queue
import re
import sys
import time
import threading
import numpy as np
import sounddevice as sd
from typing import Iterator, Optional, List, Dict, Tuple
from mlx_qwen3_asr.session import Session as ASRSession
from kokoro import KPipeline
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
SYSTEM_PROMPT       = os.environ.get("SYSTEM_PROMPT",
    "You are Jully, my AI assistant. "
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

# Qwen3-ASR STT (mlx-qwen3-asr, Apple Silicon)
# Options: "Qwen/Qwen3-ASR-0.6B" (fast) | "Qwen/Qwen3-ASR-1.7B" (accurate)
QWEN_MODEL          = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")

# Kokoro TTS
KOKORO_LANG         = "a"        # 'a' = American English, 'b' = British English
KOKORO_VOICE        = "af_heart" # af_heart | af_bella | am_adam | bf_emma …
KOKORO_SPEED        = 1.0
KOKORO_SAMPLE_RATE  = 24000      # Kokoro outputs at 24 kHz

# Chunking — how aggressively to split the LLM stream for early playback
# Splits at sentence-ending punctuation followed by whitespace.
# Add ',' to SPLIT_PATTERN for even lower latency (more, shorter chunks).
SPLIT_PATTERN       = re.compile(r'(?<=[.!?;])\s+')

# LLM streaming timeouts (seconds)
LLM_FIRST_TOKEN_TIMEOUT = 30   # give up if no token arrives within this long
LLM_INTER_TOKEN_TIMEOUT = 15   # give up if gap between tokens exceeds this

# Conversation context & memory (env-tunable)
MAX_HISTORY_PAIRS       = max(1, int(os.environ.get("VOICE_MAX_HISTORY_PAIRS", "24")))
MAX_CONTEXT_CHARS       = int(os.environ.get("VOICE_MAX_CONTEXT_CHARS", "10000"))  # 0 = no char cap
MAX_MEMORY_NOTES_CHARS  = int(os.environ.get("VOICE_MEMORY_NOTES_MAX_CHARS", "4000"))
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

def transcribe(audio: np.ndarray, model: ASRSession) -> str:
    result = model.transcribe((audio, SAMPLE_RATE), language="en")
    return result.text.strip()


# ── LLM (streaming) ────────────────────────────────────────────────────────────

def stream_llm(
    messages: List[Dict],
    api_key: str,
    model: str,
    *,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Iterator[str]:
    """Yield text tokens from a streaming OpenRouter chat completion.

    Runs the HTTP fetch in a daemon thread so requests' per-read socket
    timeout cannot cause an uninterruptible hang in the caller.
    """
    token_q: "queue.Queue[object]" = queue.Queue()

    def _fetch() -> None:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://voice-agent.local",
                    "X-Title": "Voice Agent",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True,
                },
                stream=True,
                timeout=30,
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
                        obj   = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content") or ""
                        if delta:
                            token_q.put(delta)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
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


def speak_chunked(sentence_iter: Iterator[str], pipeline: KPipeline) -> str:
    """
    Pipeline TTS synthesis with playback so they overlap:
      synth thread processes sentence N+1 while main thread plays sentence N.

    Returns the full assistant reply text for conversation history.
    """
    audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=3)
    reply_parts: List[str] = []

    def synth_worker() -> None:
        try:
            for sentence in sentence_iter:
                reply_parts.append(sentence)
                print(sentence, end=" ", flush=True)
                chunks: List[np.ndarray] = []
                for _, _, audio in pipeline(sentence, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
                    if audio is not None:
                        chunks.append(_to_numpy(audio))
                if chunks:
                    audio_q.put(np.concatenate(chunks))
        except Exception as e:
            print(f"\n[TTS error]: {e}", flush=True)
        finally:
            audio_q.put(None)  # sentinel — always sent, even on exception

    print("[Assistant]: ", end="", flush=True)
    synth_thread = threading.Thread(target=synth_worker, daemon=True)
    synth_thread.start()

    while True:
        try:
            chunk_audio = audio_q.get(timeout=60)
        except queue.Empty:
            break
        if chunk_audio is None:
            break
        sd.play(chunk_audio, samplerate=KOKORO_SAMPLE_RATE, blocking=True)

    synth_thread.join(timeout=5)
    print(flush=True)
    return " ".join(reply_parts)


def speak(text: str, pipeline: KPipeline) -> None:
    """Simple (non-streaming) TTS for greetings / one-liners."""
    print(f"[Assistant]: {text}", flush=True)
    chunks: List[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        if audio is not None:
            chunks.append(_to_numpy(audio))
    if chunks:
        sd.play(np.concatenate(chunks), samplerate=KOKORO_SAMPLE_RATE, blocking=True)


# ── Conversation helpers ───────────────────────────────────────────────────────

MEMORY_HEADER = "Earlier in this session (compressed):"


def _rest_message_chars(rest: List[Dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in rest)


def _one_line(text: str, limit: int = 200) -> str:
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 3)] + "..."


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
) -> Tuple[List[Dict], str]:
    """
    Keep up to `max_pairs` recent user/assistant turns and optionally cap total
    characters in those turns. Oldest removed pairs are appended to `memory_notes`
    (bounded) and re-injected via the system message so the model keeps coarse recall.
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

    if dropped_pairs:
        print(
            f"[Memory]: compressed {dropped_pairs} older turn pair(s); "
            f"keeping {len(rest) // 2} verbatim turn(s) in context.",
            flush=True,
        )

    out = [build_system_message(base_system_prompt, notes)] + rest
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not OPENROUTER_API_KEY:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'"
        )

    if VOICE_LIST_DEVICES:
        print_input_device_list()

    if VOICE_INPUT_DEVICE_RAW:
        try:
            apply_input_device(resolve_input_device_id(VOICE_INPUT_DEVICE_RAW))
        except ValueError as exc:
            sys.exit(f"Error: {exc}")

    print(f"Using audio input device: {describe_default_input()}", flush=True)

    print(f"Loading Qwen3-ASR ({QWEN_MODEL}) ...", flush=True)
    asr = ASRSession(QWEN_MODEL)

    print("Loading Kokoro TTS ...", flush=True)
    kokoro = KPipeline(lang_code=KOKORO_LANG)

    if MEMORY_FILE_PATH:
        conversation, memory_notes = load_memory_file(MEMORY_FILE_PATH, SYSTEM_PROMPT)
    else:
        memory_notes = ""
        conversation = [build_system_message(SYSTEM_PROMPT, memory_notes)]

    print(f"\n=== Voice Agent Ready === (model: {OPENROUTER_MODEL})", flush=True)
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
    if MEMORY_FILE_PATH:
        print(f"Memory file: {MEMORY_FILE_PATH!r}", flush=True)
    print(f"LLM max_tokens: {LLM_MAX_TOKENS}", flush=True)
    print("Press Ctrl+C to quit.\n", flush=True)
    speak("Hello! I'm ready. How can I help you?", kokoro)
    wait_after_playback()

    while True:
        try:
            # 1. Record
            audio, record_status = record_until_silence()
            if audio is None:
                print(f"[Listen skipped]: {record_status}", flush=True)
                continue

            # 2. STT
            print("[Transcribing ...]", flush=True)
            user_text = transcribe(audio, asr)
            if not user_text:
                print("[No speech detected, listening again]", flush=True)
                continue
            print(f"[You]: {user_text}", flush=True)

            # 3. Stream LLM → sentence chunks → overlapped synth+play
            conversation.append({"role": "user", "content": user_text})
            print("[Thinking ...]", flush=True)
            try:
                token_stream   = stream_llm(
                    conversation,
                    OPENROUTER_API_KEY,
                    OPENROUTER_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                )
                sentence_iter  = iter_sentences(token_stream)
                reply          = speak_chunked(sentence_iter, kokoro)
            except requests.HTTPError as e:
                print(f"[LLM Error]: {e}", flush=True)
                conversation.pop()
                continue

            conversation.append({"role": "assistant", "content": reply})
            conversation, memory_notes = trim_history_with_memory(
                conversation,
                memory_notes,
                base_system_prompt=SYSTEM_PROMPT,
                max_pairs=MAX_HISTORY_PAIRS,
                max_context_chars=MAX_CONTEXT_CHARS,
            )
            if MEMORY_FILE_PATH:
                save_memory_file(MEMORY_FILE_PATH, memory_notes, conversation)
            wait_after_playback()

        except KeyboardInterrupt:
            print("\n\nGoodbye!", flush=True)
            if MEMORY_FILE_PATH:
                save_memory_file(MEMORY_FILE_PATH, memory_notes, conversation)
            speak("Goodbye!", kokoro)
            break
        except Exception as e:
            print(f"[Unexpected error]: {e}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
