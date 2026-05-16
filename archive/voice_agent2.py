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
from typing import Iterator, Optional, List, Dict
from mlx_qwen3_asr.session import Session as ASRSession
from kokoro import KPipeline
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
SYSTEM_PROMPT       = (
    "You are Molly, my beautiful, sex starved, nimphomaniac girlfriend. "
    "Keep responses concise and conversational — 1-3 sentences unless more detail is asked for. "
    "Avoid emojis, markdown, bullet points, or special characters since your output will be spoken aloud."
)

# Audio capture
SAMPLE_RATE         = 16000   # Qwen3-ASR expects 16 kHz
CHANNELS            = 1
BLOCK_SIZE          = 512
SILENCE_THRESHOLD   = 0.02    # RMS amplitude below this = silence
SILENCE_SECS        = 1.5     # Stop recording after this many silent seconds
MIN_SPEECH_SECS     = 0.4     # Ignore clips shorter than this
MAX_RECORD_SECS     = 30      # Hard cap to avoid infinite recording

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
# ───────────────────────────────────────────────────────────────────────────────


# ── Audio recording ────────────────────────────────────────────────────────────

def record_until_silence() -> Optional[np.ndarray]:
    """Stream mic input; return float32 array when silence follows speech."""
    print("\n[Listening ...]", flush=True)

    audio_chunks: List[np.ndarray] = []
    speech_chunks = 0
    silent_chunks = 0
    has_speech    = False
    stop_event    = threading.Event()

    silence_limit = int(SILENCE_SECS * SAMPLE_RATE / BLOCK_SIZE)
    min_speech    = int(MIN_SPEECH_SECS * SAMPLE_RATE / BLOCK_SIZE)

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, speech_chunks, has_speech
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
            stop_event.set()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        blocksize=BLOCK_SIZE, dtype="float32", callback=callback,
    ):
        stop_event.wait(timeout=MAX_RECORD_SECS)

    if not has_speech or speech_chunks < min_speech:
        return None
    return np.concatenate(audio_chunks)


# ── STT ────────────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray, model: ASRSession) -> str:
    result = model.transcribe((audio, SAMPLE_RATE), language="en")
    return result.text.strip()


# ── LLM (streaming) ────────────────────────────────────────────────────────────

def stream_llm(messages: List[Dict], api_key: str, model: str) -> Iterator[str]:
    """Yield text tokens from a streaming OpenRouter chat completion."""
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
            "max_tokens": 300,
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
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


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

    synth_thread.join()
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

def trim_history(messages: List[Dict], max_pairs: int = 10) -> List[Dict]:
    system = [m for m in messages if m["role"] == "system"]
    rest   = [m for m in messages if m["role"] != "system"]
    return system + rest[-(max_pairs * 2):]


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not OPENROUTER_API_KEY:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'"
        )

    print(f"Loading Qwen3-ASR ({QWEN_MODEL}) ...", flush=True)
    asr = ASRSession(QWEN_MODEL)

    print("Loading Kokoro TTS ...", flush=True)
    kokoro = KPipeline(lang_code=KOKORO_LANG)

    conversation: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"\n=== Voice Agent Ready === (model: {OPENROUTER_MODEL})", flush=True)
    print("Press Ctrl+C to quit.\n", flush=True)
    speak("Hello! I'm ready. How can I help you?", kokoro)

    while True:
        try:
            # 1. Record
            audio = record_until_silence()
            if audio is None:
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
                token_stream   = stream_llm(conversation, OPENROUTER_API_KEY, OPENROUTER_MODEL)
                sentence_iter  = iter_sentences(token_stream)
                reply          = speak_chunked(sentence_iter, kokoro)
            except requests.HTTPError as e:
                print(f"[LLM Error]: {e}", flush=True)
                conversation.pop()
                continue

            conversation.append({"role": "assistant", "content": reply})
            conversation = trim_history(conversation)

        except KeyboardInterrupt:
            print("\n\nGoodbye!", flush=True)
            speak("Goodbye!", kokoro)
            break
        except Exception as e:
            print(f"[Unexpected error]: {e}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
