# ws_fixed.py
import os
import asyncio
import logging
from urllib.parse import parse_qs

from fastapi import WebSocket
import azure.cognitiveservices.speech as speechsdk
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# =============================
# Logging
# =============================
log = logging.getLogger("asr.fixed")
log.setLevel(logging.INFO)

# =============================
# Language map (fixed mode)
# =============================
LANG_MAP = {
    "english": "en-US", "en": "en-US",
    "chinese": "zh-TW", "mandarin": "zh-CN", "zh": "zh-CN",
    "japanese": "ja-JP", "ja": "ja-JP",
    "korean": "ko-KR", "ko": "ko-KR",
    "thai": "th-TH", "th": "th-TH",
    "vietnamese": "vi-VN", "vi": "vi-VN",
    "indonesian": "id-ID", "id": "id-ID",
    "malay": "ms-MY", "ms": "ms-MY",
    "hindi": "hi-IN", "hi": "hi-IN",
    "french": "fr-FR", "fr": "fr-FR",
    "german": "de-DE", "de": "de-DE",
    "spanish": "es-ES", "es": "es-ES",
    "portuguese": "pt-PT", "pt": "pt-PT",
}

# =============================
# Translation config
# =============================
TARGET_LANG = "en"
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")

# =============================
# Azure Translator
# =============================
async def azure_translate(text: str, target: str) -> str:
    if not text or not text.strip():
        return ""

    url = "https://api.cognitive.microsofttranslator.com/translate"
    params = {
        "api-version": "3.0",
        "to": target,
    }
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                params=params,
                headers=headers,
                json=[{"text": text}],
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    log.error(f"❌ Azure translate HTTP {resp.status}: {err}")
                    return ""

                data = await resp.json()
                return data[0]["translations"][0]["text"]
    except Exception as e:
        log.error(f"❌ translate exception: {e}")
        return ""

# =============================
# WebSocket: Fixed Language Mode
# =============================
async def ws_fixed(ws: WebSocket):
    """
    Fixed language ASR pipeline
    ✅ instant ASR
    ✅ instant partial translation
    ✅ final translation
    ❌ no Whisper
    ❌ no detect / rollback
    """
    await ws.accept()

    # -------- parse ?lang= --------
    query = parse_qs(ws.url.query)
    lang = query.get("lang", ["en"])[0].lower()

    azure_lang = LANG_MAP.get(lang)
    if not azure_lang:
        log.warning(f"Unknown lang '{lang}', fallback to en")
        lang = "en"
        azure_lang = LANG_MAP["en"]

    log.info(f"🔒 Fixed mode start: {lang} → {azure_lang}")

    # -------- notify frontend --------
    await ws.send_json({
        "type": "lang_locked",
        "lang": lang,
    })

    # -------- Azure Speech setup --------
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        raise RuntimeError("AZURE SPEECH ENV NOT SET")

    cfg = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region,
    )
    cfg.speech_recognition_language = azure_lang

    push_stream = speechsdk.audio.PushAudioInputStream(
        speechsdk.audio.AudioStreamFormat(16000, 16, 1)
    )

    recognizer = speechsdk.SpeechRecognizer(
        cfg,
        speechsdk.audio.AudioConfig(stream=push_stream),
    )

    loop = asyncio.get_running_loop()

    # =============================
    # Partial translation state
    # =============================
    last_partial_text = ""
    translate_task: asyncio.Task | None = None

    # -----------------------------
    # helpers
    # -----------------------------
    async def send_partial(text: str):
        await ws.send_json({"type": "partial", "text": text})

    async def send_mid_translate(text: str):
        if not text.strip():
            return
        trans = await azure_translate(text, TARGET_LANG)
        if trans:
            await ws.send_json({
                "type": "mid_translate",
                "translated": trans,
            })

    async def send_final(text: str):
        log.info(f"📝 ASR FINAL: {text}")

        await ws.send_json({
            "type": "final",
            "text": text,
        })

        trans = await azure_translate(text, TARGET_LANG)
        log.info(f"🌍 FINAL TRANSLATED: {trans}")

        if trans:
            await ws.send_json({
                "type": "final_translate",
                "translated": trans,
                "provisional": False,
                "replace_last": False,
            })

        # 清掉 partial 翻譯
        await ws.send_json({
            "type": "mid_translate",
            "translated": "",
        })

    # -----------------------------
    # callbacks
    # -----------------------------
    def on_recognizing(evt):
        nonlocal last_partial_text, translate_task

        text = evt.result.text
        if not text:
            return

        # partial ASR
        asyncio.run_coroutine_threadsafe(
            send_partial(text),
            loop,
        )

        # 沒變就不翻
        if text == last_partial_text:
            return
        last_partial_text = text

        # cancel 上一次翻譯
        if translate_task:
            translate_task.cancel()

        async def delayed_translate():
            try:
                await asyncio.sleep(0.3)  # debounce
                await send_mid_translate(text)
            except asyncio.CancelledError:
                pass

        translate_task = loop.create_task(delayed_translate())

    def on_recognized(evt):
        if evt.result.text:
            asyncio.run_coroutine_threadsafe(
                send_final(evt.result.text),
                loop,
            )

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)

    recognizer.start_continuous_recognition_async().get()
    log.info("🟢 Azure ASR started (fixed + instant translate)")

    # -------- audio loop --------
    try:
        async for chunk in ws.iter_bytes():
            push_stream.write(chunk)

    finally:
        log.info("🔌 Fixed client disconnected")
        try:
            recognizer.stop_continuous_recognition_async().get()
            push_stream.close()
        except Exception:
            pass
