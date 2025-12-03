import os
import io
import time
import asyncio
import struct
import logging
from typing import List, Dict, Any, Optional

import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from ws_fixed import ws_fixed
from ws_multilang_adaptive import ws_multilang_adaptive


# =============================
# Init
# =============================
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asr")

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
  raise RuntimeError("AZURE SPEECH ENV NOT SET")
if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_REGION:
  raise RuntimeError("AZURE TRANSLATOR ENV NOT SET")
if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY NOT SET")

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

# 目標翻譯語言（Azure Translator 的 to）
TARGET_LANG = "en"

# =============================
# Language map (Whisper -> Azure)
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
# Detection timing
# =============================
SILENCE_RMS_THRESHOLD = 300
MIN_DETECT_SEC = 0.8   # 等 0.8 秒就先用 Whisper 判語言
MAX_DETECT_SEC = 8.0

# =============================
# Utils
# =============================
def pcm_to_wav(pcm: bytes) -> bytes:
  """16 kHz mono PCM16 -> WAV bytes"""
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
# Whisper detect (language only)
# =============================
async def whisper_detect(pcm: bytes) -> str:
  """用 Whisper 根據音檔判斷語言（不取文字）"""
  if not pcm:
    return "unknown"

  wav = pcm_to_wav(pcm)
  url = "https://api.openai.com/v1/audio/transcriptions"
  headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

  form = aiohttp.FormData()
  form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
  form.add_field("model", "whisper-1")
  form.add_field("response_format", "verbose_json")

  async with aiohttp.ClientSession() as session:
    async with session.post(url, headers=headers, data=form) as resp:
      if resp.status != 200:
        log.error(await resp.text())
        return "unknown"
      data = await resp.json()
      return (data.get("language") or "unknown").lower()


# =============================
# Whisper transcribe (原文)
# =============================
async def whisper_transcribe(pcm: bytes) -> str:
  """用 Whisper 拿「正確原文」（不翻譯）"""
  if not pcm:
    return ""
  wav = pcm_to_wav(pcm)
  url = "https://api.openai.com/v1/audio/transcriptions"
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
# Whisper translate (for correction)
# =============================
async def whisper_translate(pcm: bytes) -> str:
  """
  用 Whisper 直接把這個 utterance 翻成 TARGET_LANG
  （專門用在「語言判錯後」的補救）
  """
  if not pcm:
    return ""
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
# Azure Translator
# =============================
async def azure_translate(text: str, target: str) -> str:
  if not text.strip():
    return ""
  url = "https://api.cognitive.microsofttranslator.com/translate"
  params = {"api-version": "3.0", "to": target}
  headers = {
    "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
    "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
    "Content-Type": "application/json",
  }
  async with aiohttp.ClientSession() as session:
    async with session.post(
      url, params=params, headers=headers, json=[{"text": text}]
    ) as resp:
      data = await resp.json()
      return data[0]["translations"][0]["text"]


# =============================
# Embedding & cosine for merge
# =============================
async def embed(text: str):
  if not text:
    return None
  url = "https://api.openai.com/v1/embeddings"
  headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
  async with aiohttp.ClientSession() as session:
    async with session.post(
      url,
      headers=headers,
      json={"model": "text-embedding-3-small", "input": text},
    ) as resp:
      if resp.status != 200:
        log.error(await resp.text())
        return None
      data = await resp.json()
      return data["data"][0]["embedding"]


def cosine(a, b) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  na = sum(x * x for x in a) ** 0.5
  nb = sum(x * x for x in b) ** 0.5
  return dot / (na * nb + 1e-8)


# =============================
# WebSocket ASR
# =============================
@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
  await ws.accept()
  log.info("🔌 client connected")

  # --- audio buffers ---
  detect_buffer = bytearray()         # 語言偵測用（前 0.8s）
  current_utt_audio = bytearray()     # 當前 utterance 的 PCM（給 Whisper 保險用）

  # --- language state ---
  provisional_lang: Optional[str] = None  # Whisper 第一輪判到的語言（還沒保險）
  lang_locked = False                     # 是否已經「第一句驗證通過」→ lock in

  recognizer: Optional[speechsdk.SpeechRecognizer] = None
  push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
  loop = asyncio.get_running_loop()

  first_speech_time: Optional[float] = None

  # --- sentence merge state (for translation output) ---
  last_translation_text: Optional[str] = None
  last_translation_time: Optional[float] = None
  last_translation_embedding: Any = None

  # --- utterance history (for rewind when mismatch) ---
  # 每個元素：{"pcm": bytes, "azure_text": str, "azure_translation": str}
  utterance_history: List[Dict[str, Any]] = []

  # =============================
  # Azure ASR control
  # =============================
  async def start_azure(lang: str):
    """啟動指定語言的 Azure ASR session"""
    nonlocal recognizer, push_stream
    nonlocal last_translation_text, last_translation_time, last_translation_embedding

    azure_lang = LANG_MAP.get(lang, "en-US")
    log.info(f"🟢 Azure ASR start: {azure_lang}")

    cfg = speechsdk.SpeechConfig(
      subscription=AZURE_SPEECH_KEY,
      region=AZURE_SPEECH_REGION,
    )
    cfg.speech_recognition_language = azure_lang

    push_stream = speechsdk.audio.PushAudioInputStream(
      speechsdk.audio.AudioStreamFormat(16000, 16, 1)
    )
    recognizer = speechsdk.SpeechRecognizer(
      cfg, speechsdk.audio.AudioConfig(stream=push_stream)
    )

    async def send_mid_translate(text: str):
      # partial 的即時翻譯：語言還沒 lock 的時候也可以翻，但前端可以標示「provisional」
      trans = await azure_translate(text, TARGET_LANG)
      await ws.send_json({"type": "mid_translate", "translated": trans})

    def on_partial(evt):
      if evt.result.text:
        # partial ASR
        asyncio.run_coroutine_threadsafe(
          ws.send_json({"type": "partial", "text": evt.result.text}),
          loop,
        )
        # partial 翻譯
        asyncio.run_coroutine_threadsafe(
          send_mid_translate(evt.result.text),
          loop,
        )

    async def on_final(text: str):
      """
      每個 utterance 結束時被呼叫：
      1. 先送 Azure ASR 原文 & 翻譯（provisional / final）
      2. 再拿這個 utterance 的 PCM 給 Whisper 重判語言做「保險」
      3. 若發現 mismatch → 回溯「所有已經講過的 utterances」重新翻譯
      """
      nonlocal provisional_lang, lang_locked
      nonlocal current_utt_audio
      nonlocal last_translation_text, last_translation_time, last_translation_embedding
      nonlocal utterance_history

      # 這個 utterance 單獨的 PCM（保險用）
      utt_pcm = bytes(current_utt_audio)

      # 先把這一句記錄到 history（先記 PCM，之後補上 text / translation）
      history_entry: Dict[str, Any] = {
        "pcm": utt_pcm,
        "azure_text": text,
        "azure_translation": None,
      }
      utterance_history.append(history_entry)

      # -----------------
      # 1) 原文
      # -----------------
      await ws.send_json({"type": "final", "text": text})

      # 2) 翻譯（先照目前語言翻，是否 provisional 由 lang_locked 決定）
      trans = await azure_translate(text, TARGET_LANG)
      history_entry["azure_translation"] = trans

      now = time.perf_counter()
      merge = False
      emb = None
      try:
        emb = await embed(trans)
      except Exception as e:
        log.error(f"embedding error: {e}")

      if (
        emb is not None
        and last_translation_embedding is not None
        and last_translation_time is not None
      ):
        sim = cosine(emb, last_translation_embedding)
        dt = now - last_translation_time
        # 語境接近 / 停頓短 → 視為同一句補尾巴
        if sim > 0.75 and dt < 1.0:
          merge = True

      if merge and last_translation_text is not None:
        merged_text = last_translation_text + " " + trans
        await ws.send_json(
          {
            "type": "final_translate",
            "translated": merged_text,
            "provisional": not lang_locked,
            "replace_last": True,
          }
        )
        last_translation_text = merged_text
      else:
        await ws.send_json(
          {
            "type": "final_translate",
            "translated": trans,
            "provisional": not lang_locked,
            "replace_last": False,
          }
        )
        last_translation_text = trans

      if emb is not None:
        last_translation_embedding = emb
        last_translation_time = now

      # -----------------
      # 3) Whisper 保險：用「整句 utterance」重判語言
      # -----------------
      detected = await whisper_detect(utt_pcm)
      log.info(f"🔍 Verification detect (this utt): {detected}")

      # 第一次有 provisional 結果時，Whisper 也還沒判過 → 這裡補上
      if provisional_lang is None and detected in LANG_MAP:
        provisional_lang = detected

      # ✅ 語言已經 lock 過就不再動了（你目前設計）
      if lang_locked:
        current_utt_audio.clear()
        return

      # --- Case A: mismatch → 直接改用新語言，並回溯所有 utterances ---
      if provisional_lang is not None and detected in LANG_MAP and detected != provisional_lang:
        log.info(
          f"⚠️ Utterance lang mismatch: provisional={provisional_lang}, detected={detected}"
        )

        # 0) reset translation merge context
        last_translation_text = None
        last_translation_embedding = None
        last_translation_time = None

        # 1) 通知前端：所有之前的 ASR / translation 都是錯的 → 全部畫刪除線
        await ws.send_json({"type": "invalidate_all_asr"})
        await ws.send_json({"type": "invalidate_all_translation"})

        # 2) 用新語言（實際上 Whisper auto detect）對「所有歷史 utterances」重翻
        for idx, entry in enumerate(utterance_history):
          pcm_bytes: bytes = entry["pcm"]

          correct_text = await whisper_transcribe(pcm_bytes)
          correct_trans = await whisper_translate(pcm_bytes)

          if correct_text.strip():
            await ws.send_json(
              {
                "type": "final",
                "text": correct_text,
                "corrected": True,
                "replayed": True,
              }
            )

          if correct_trans.strip():
            await ws.send_json(
              {
                "type": "final_translate",
                "translated": correct_trans,
                "provisional": False,
                "replace_last": False,
                "replayed": True,
              }
            )
            # 更新 merge context（以最後一條為基準）
            now2 = time.perf_counter()
            try:
              emb2 = await embed(correct_trans)
            except Exception as e:
              log.error(f"embedding(corrected) error: {e}")
              emb2 = None
            last_translation_text = correct_trans
            if emb2 is not None:
              last_translation_embedding = emb2
              last_translation_time = now2

        # 3) 清掉 history（因為已經用 Whisper 正確重放）
        utterance_history.clear()

        # 4) 把語言 lock 在新的 detected，並重啟 Azure ASR
        provisional_lang = detected
        lang_locked = True
        await ws.send_json({"type": "lang_locked", "lang": detected})
        await restart_azure()

      # --- Case B: match → 第一句驗證通過，直接 lock in ---
      elif provisional_lang is not None and detected == provisional_lang:
        log.info(f"✅ Language verified and locked: {detected}")
        lang_locked = True
        await ws.send_json({"type": "lang_locked", "lang": detected})

      # （Case C: detected 不在 LANG_MAP or unknown → 先保持現狀，等下一句再說）

      current_utt_audio.clear()

    recognizer.recognizing.connect(on_partial)
    recognizer.recognized.connect(
      lambda e: asyncio.run_coroutine_threadsafe(
        on_final(e.result.text), loop
      )
    )
    recognizer.start_continuous_recognition_async().get()

  async def restart_azure():
    """停掉舊的 recognizer，換新的語言重啟"""
    nonlocal recognizer, push_stream
    log.info("♻️ Restart Azure ASR with new language")

    try:
      if recognizer:
        recognizer.stop_continuous_recognition_async().get()
      if push_stream:
        push_stream.close()
    except Exception:
      pass

    recognizer = None
    push_stream = None

    # 給 Azure SDK 一點時間確實停乾淨
    await asyncio.sleep(0.2)

    if provisional_lang:
      await start_azure(provisional_lang)

  # =============================
  # Main loop
  # =============================
  try:
    async for chunk in ws.iter_bytes():
      # chunk = 16kHz mono PCM16
      current_utt_audio.extend(chunk)

      # --- 語言尚未決定，先做偵測（UNTIL） ---
      if provisional_lang is None:
        # 太小聲就先丟掉
        if rms_energy(chunk) < SILENCE_RMS_THRESHOLD:
          continue

        if first_speech_time is None:
          first_speech_time = time.perf_counter()

        detect_buffer.extend(chunk)
        elapsed = time.perf_counter() - first_speech_time

        # 至少聽滿 MIN_DETECT_SEC 再丟去 Whisper
        if elapsed >= MIN_DETECT_SEC and provisional_lang is None:
          lang = await whisper_detect(bytes(detect_buffer))
          log.info(f"🕒 Initial detect language: {lang}")
          if lang not in LANG_MAP:
            lang = "english"
          provisional_lang = lang

        # 最長不能超過 MAX_DETECT_SEC，超過就硬用英文
        if elapsed >= MAX_DETECT_SEC and provisional_lang is None:
          provisional_lang = "english"

        # 一旦決定 provisional_lang：啟動 Azure ASR，開始即時翻譯
        if provisional_lang:
          await ws.send_json({"type": "lang", "lang": provisional_lang})
          await start_azure(provisional_lang)
          if push_stream and detect_buffer:
            push_stream.write(bytes(detect_buffer))
          detect_buffer.clear()
        continue

      # --- 語言已經有 provisional，直接塞到 Azure push_stream ---
      if push_stream:
        push_stream.write(chunk)

  finally:
    log.info("🔌 client disconnected")
    try:
      if recognizer:
        recognizer.stop_continuous_recognition_async().get()
      if push_stream:
        push_stream.close()
    except Exception:
      pass


@app.websocket("/ws/fixed")
async def ws_fixed_entry(ws: WebSocket):
    await ws_fixed(ws)
    

@app.websocket("/ws/multilang")
async def multilang_endpoint(ws: WebSocket):
    await ws_multilang_adaptive(ws)