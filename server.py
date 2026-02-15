"""
Omi × Groq Whisper STT ブリッジサーバー
======================================
Omiウェアラブルからの音声をWebSocketで受け取り、
Groq APIのWhisper Large-v3で文字起こしして返す中継サーバー。
"""

import asyncio
import io
import json
import logging
import os
import struct
import time
import wave
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ============================================================
#  設定（環境変数で変更可能）
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
LANGUAGE = os.environ.get("LANGUAGE", "ja")
PROMPT = os.environ.get("PROMPT", "日本語の会話です")
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omi-groq-stt")

app = FastAPI(title="Omi-Groq STT Bridge")


# ============================================================
#  OGG Opus コンテナ作成（opus生フレーム → Groqが受け付けるoggファイル）
# ============================================================

# OGG CRC-32 ルックアップテーブル（OGG独自の多項式 0x04C11DB7）
_CRC_TABLE = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if _r & 0x80000000 else (_r << 1) & 0xFFFFFFFF
    _CRC_TABLE.append(_r)


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((_CRC_TABLE[((crc >> 24) ^ b) & 0xFF]) ^ (crc << 8)) & 0xFFFFFFFF
    return crc


def _make_ogg_page(
    serial: int, page_seq: int, granule: int, data: bytes,
    bos: bool = False, eos: bool = False,
) -> bytes:
    """1つのOGGページを作成"""
    header_type = 0
    if bos:
        header_type |= 0x02
    if eos:
        header_type |= 0x04

    # セグメントテーブル（255バイト単位で分割）
    segments = []
    remaining = len(data)
    while remaining >= 255:
        segments.append(255)
        remaining -= 255
    segments.append(remaining)

    # ヘッダー（CRC=0 のプレースホルダー）
    header = struct.pack(
        "<4sBBqIIIB",
        b"OggS", 0, header_type, granule,
        serial, page_seq, 0, len(segments),
    )
    header += bytes(segments)

    # CRC計算してヘッダーに埋め込む
    full_page_no_crc = header + data
    crc = _ogg_crc(full_page_no_crc)
    header = header[:22] + struct.pack("<I", crc) + header[26:]

    return header + data


def create_ogg_opus(
    opus_frames: list, sample_rate: int = 16000, frame_duration_ms: int = 20,
) -> bytes:
    """opusの生フレーム群からOGG Opusファイルを作成"""
    serial = 1
    page_seq = 0
    pages = []

    # OpusHead パケット
    opus_head = struct.pack(
        "<8sBBHIhB",
        b"OpusHead", 1, 1, 3840, sample_rate, 0, 0,
    )
    pages.append(_make_ogg_page(serial, page_seq, 0, opus_head, bos=True))
    page_seq += 1

    # OpusTags パケット
    vendor = b"omi-groq-stt"
    opus_tags = (
        struct.pack("<8sI", b"OpusTags", len(vendor))
        + vendor
        + struct.pack("<I", 0)
    )
    pages.append(_make_ogg_page(serial, page_seq, 0, opus_tags))
    page_seq += 1

    # オーディオページ（各フレームを1ページに格納）
    samples_per_frame = sample_rate * frame_duration_ms // 1000
    granule = 0

    for i, frame in enumerate(opus_frames):
        granule += samples_per_frame
        is_last = i == len(opus_frames) - 1
        pages.append(_make_ogg_page(serial, page_seq, granule, frame, eos=is_last))
        page_seq += 1

    return b"".join(pages)


# ============================================================
#  PCM → WAV 変換
# ============================================================

def pcm_to_wav(
    pcm_data: bytes, sample_rate: int = 16000,
    sample_width: int = 2, channels: int = 1,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)
    return buf.getvalue()


def pcm8_to_pcm16(data: bytes) -> bytes:
    """8-bit unsigned PCM → 16-bit signed PCM"""
    result = bytearray(len(data) * 2)
    for i, b in enumerate(data):
        struct.pack_into("<h", result, i * 2, (b - 128) << 8)
    return bytes(result)


# ============================================================
#  Whisperハルシネーション（幻覚）フィルター
# ============================================================

HALLUCINATION_PATTERNS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe to",
    "see you next time",
    "see you in the next",
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "お楽しみに",
    "시청해 주셔서 감사합니다",
    "구독",
    "Спасибо за просмотр",
    "Подписывайтесь",
    "MBC 뉴스",
    "Amara.org",
    "www.",
    "http",
]


def is_hallucination(text: str) -> bool:
    """Whisperのよくある幻覚パターンを検出"""
    t = text.strip().lower()

    if len(t) < 2:
        return True

    for pattern in HALLUCINATION_PATTERNS:
        if pattern.lower() in t:
            return True

    # 同じ短いフレーズが繰り返されるパターン
    if len(t) < 30:
        words = t.split()
        if len(words) >= 4 and len(set(words)) <= 2:
            return True

    return False


# ============================================================
#  Groq API 呼び出し
# ============================================================

async def transcribe_groq(audio_data: bytes, filename: str = "audio.wav") -> list:
    """Groq Whisper APIで文字起こし"""
    from groq import Groq

    try:
        client = Groq(api_key=GROQ_API_KEY)

        transcription = await asyncio.to_thread(
            client.audio.transcriptions.create,
            file=(filename, audio_data),
            model=WHISPER_MODEL,
            language=LANGUAGE,
            temperature=0.0,
            prompt=PROMPT,
            response_format="verbose_json",
        )

        segments = []

        if hasattr(transcription, "segments") and transcription.segments:
            for seg in transcription.segments:
                # 無音確率が高いセグメントはスキップ
                if seg.get("no_speech_prob", 0) > 0.7:
                    continue
                text = seg.get("text", "").strip()
                if not text or is_hallucination(text):
                    continue
                segments.append({
                    "text": text,
                    "speaker": "SPEAKER_00",
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                })

        elif hasattr(transcription, "text") and transcription.text.strip():
            text = transcription.text.strip()
            if not is_hallucination(text):
                segments.append({
                    "text": text,
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 0.0,
                })

        logger.info(f"Transcribed: {len(segments)} segments")
        return segments

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return []


# ============================================================
#  WebSocket ハンドラー（Omiからの接続を受け付け）
# ============================================================

@app.websocket("/listen")
async def listen(websocket: WebSocket):
    await websocket.accept()

    # Omiが送ってくるクエリパラメータを読み取る
    codec = websocket.query_params.get("codec", "pcm8")
    sample_rate = int(websocket.query_params.get("sample_rate", "16000"))

    logger.info(f"Omi connected: codec={codec}, sample_rate={sample_rate}")

    is_opus = "opus" in codec.lower()

    # バッファ
    pcm_buffer = bytearray()
    opus_frames: list = []
    last_send = time.time()
    segment_offset = 0.0  # 文字起こしのタイムスタンプオフセット

    async def flush_buffer() -> list:
        """バッファの音声をGroqに送って文字起こし結果を返す"""
        nonlocal pcm_buffer, opus_frames, last_send, segment_offset

        segments = []
        chunk_duration = 0.0

        if is_opus and opus_frames:
            frame_duration_ms = 20  # Omiデバイスの一般的なフレームサイズ
            chunk_duration = len(opus_frames) * frame_duration_ms / 1000.0
            ogg_data = create_ogg_opus(opus_frames, sample_rate, frame_duration_ms)
            opus_frames = []
            last_send = time.time()
            segments = await transcribe_groq(ogg_data, "audio.ogg")

        elif not is_opus and len(pcm_buffer) > 0:
            if codec == "pcm8":
                pcm16_data = pcm8_to_pcm16(bytes(pcm_buffer))
            else:
                pcm16_data = bytes(pcm_buffer)
            chunk_duration = len(pcm16_data) / (sample_rate * 2)
            wav_data = pcm_to_wav(pcm16_data, sample_rate)
            pcm_buffer = bytearray()
            last_send = time.time()
            segments = await transcribe_groq(wav_data, "audio.wav")

        # タイムスタンプにオフセットを加算
        for seg in segments:
            seg["start"] += segment_offset
            seg["end"] += segment_offset

        segment_offset += chunk_duration
        return segments

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                # タイムアウト時にバッファをフラッシュするか確認
                elapsed = time.time() - last_send
                has_data = (is_opus and len(opus_frames) > 10) or (
                    not is_opus and len(pcm_buffer) > sample_rate
                )
                if elapsed >= CHUNK_SECONDS and has_data:
                    segments = await flush_buffer()
                    if segments:
                        await websocket.send_json({"segments": segments})
                continue

            if msg["type"] == "websocket.disconnect":
                break

            # バイナリデータ（音声）
            if "bytes" in msg and msg["bytes"]:
                data = msg["bytes"]
                if len(data) <= 2:  # ハートビートping
                    continue

                if is_opus:
                    opus_frames.append(data)
                else:
                    pcm_buffer.extend(data)

                # チャンク間隔を超えたらフラッシュ
                elapsed = time.time() - last_send
                has_data = (is_opus and len(opus_frames) > 10) or (
                    not is_opus and len(pcm_buffer) > sample_rate
                )
                if elapsed >= CHUNK_SECONDS and has_data:
                    segments = await flush_buffer()
                    if segments:
                        await websocket.send_json({"segments": segments})

            # テキストメッセージ（制御コマンド）
            elif "text" in msg and msg["text"]:
                try:
                    text_msg = json.loads(msg["text"])
                    if text_msg.get("type") == "CloseStream":
                        logger.info("CloseStream received")
                        segments = await flush_buffer()
                        if segments:
                            await websocket.send_json({"segments": segments})
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Omi disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("Connection closed")


# ============================================================
#  ヘルスチェック & ルート
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "model": WHISPER_MODEL, "language": LANGUAGE}


@app.get("/")
async def root():
    return {
        "service": "Omi-Groq STT Bridge",
        "description": "Omiの音声をGroq Whisperで文字起こしする中継サーバー",
        "websocket_endpoint": "/listen",
        "health": "/health",
    }
