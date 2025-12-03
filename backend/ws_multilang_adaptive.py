# ws_multilang_adaptive.py
import os
import io
import struct
import time
import asyncio
import logging
from typing import List, Optional

from fastapi import WebSocket
from dotenv import load_dotenv
import aiohttp
import azure.cognitiveservices.speech as speechsdk

# =============================
# Init
# =============================
load_dotenv()

log = logging.getLogger("asr.multilang")
log.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY NOT SET")
if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise RuntimeError("AZURE SPEECH ENV NOT SET")

TARGET_LANG = "en"

# =============================
# Language candidate cache
# =============================
LANG_CANDIDATES: List[str] = []
MAX_CANDIDATES = 5


def update_candidates(lang: str):
    if not lang or lang == "unknown":
        return
    if lang in LANG_CANDIDATES:
        LANG_CANDIDATES.remove(lang)
    LANG_CANDIDATES.insert(0, lang)
    if len(LANG_CANDIDATES) > MAX_CANDIDATES:
        LANG_CANDIDATES.pop()
    log.info(f"🌐 Language candidates: {LANG_CANDIDATES}")


def get_lang_hint() -> Optional[str]:
    return LANG_CANDIDATES[0] if LANG_CANDIDATES else None


# =============================
# Audio utils
# =============================
def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


def rms_energy(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


# =============================
# Whisper helpers
# =============================
async def whisper_transcribe_with_hint(
    pcm: bytes,
    lang_hint: Optional[str],
):
    wav = pcm_to_wav(pcm)
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    form = aiohttp.FormData()
    form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", "whisper-1")
    form.add_field("response_format", "verbose_json")

    # ✅ Fast-path: language hint (skip detect)
    if lang_hint:
        form.add_field("language", lang_hint)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form) as resp:
            if resp.status != 200:
                log.error(await resp.text())
                return "unknown", ""
            data = await resp.json()
            return (data.get("language") or "unknown"), (data.get("text") or "")


async def whisper_translate(pcm: bytes) -> str:
    wav = pcm_to_wav(pcm)
    url = "https://api.openai.com/v1/audio/translations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    form = aiohttp.FormData()
    form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", "whisper-1")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form) as resp:
            if resp.status != 200:
                log.error(await resp.text())
                return ""
            data = await resp.json()
            return data.get("text", "")


# =============================
# WebSocket entry
# =============================
async def ws_multilang_adaptive(ws: WebSocket):
    await ws.accept()
    log.info("🔌 Multi-lang adaptive connected")

    # -------- buffers --------
    current_utt_audio = bytearray()
    last_voice_time: Optional[float] = None

    # silence params (same class as auto mode)
    SILENCE_RMS = 300
    SILENCE_SEC = 0.8

    # =============================
    # Azure ASR (partial only)
    # =============================
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )

    auto_lang_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=[
            "en-US", "ja-JP", "zh-TW", "zh-CN",
            "ko-KR", "es-ES", "fr-FR", "de-DE",
            "pt-PT", "th-TH", "vi-VN", "id-ID",
            "ms-MY", "hi-IN",
        ]
    )

    audio_format = speechsdk.audio.AudioStreamFormat(16000, 16, 1)
    push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_lang_config,
        audio_config=audio_config,
    )

    loop = asyncio.get_running_loop()

    async def send_partial(text: str):
        await ws.send_json({"type": "partial", "text": text})

    def on_recognizing(evt):
        if evt.result.text:
            asyncio.run_coroutine_threadsafe(
                send_partial(evt.result.text),
                loop,
            )

    recognizer.recognizing.connect(on_recognizing)
    recognizer.start_continuous_recognition_async().get()
    log.info("🟢 Azure ASR started (partial only)")

    # =============================
    # Utterance handler
    # =============================
    async def handle_utterance(pcm: bytes):
        # ✅ Fast-path: use cached language
        lang_hint = get_lang_hint()
        lang, text = await whisper_transcribe_with_hint(pcm, lang_hint)

        # 🛡 fallback: hint decode failed
        if not text.strip():
            lang, text = await whisper_transcribe_with_hint(pcm, None)

        log.info(f"🧠 Whisper result: lang={lang}, text={text}")
        update_candidates(lang)

        if text.strip():
            await ws.send_json({
                "type": "final",
                "text": text,
                "lang": lang,
            })

        translated = await whisper_translate(pcm)
        log.info(f"🌍 Whisper translated: {translated}")

        if translated.strip():
            await ws.send_json({
                "type": "final_translate",
                "translated": translated,
                "provisional": False,
                "replace_last": False,
                "lang": lang,
            })

    # =============================
    # Main audio loop
    # =============================
    try:
        async for chunk in ws.iter_bytes():
            current_utt_audio.extend(chunk)
            push_stream.write(chunk)

            energy = rms_energy(chunk)
            now = time.perf_counter()

            if energy > SILENCE_RMS:
                last_voice_time = now
            else:
                if (
                    last_voice_time
                    and now - last_voice_time > SILENCE_SEC
                    and current_utt_audio
                ):
                    utt_pcm = bytes(current_utt_audio)
                    current_utt_audio.clear()
                    last_voice_time = None

                    asyncio.create_task(handle_utterance(utt_pcm))

    finally:
        log.info("🔌 Multi-lang adaptive disconnected")
        try:
            recognizer.stop_continuous_recognition_async().get()
            push_stream.close()
        except Exception:
            pass
