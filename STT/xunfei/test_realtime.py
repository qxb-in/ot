import websocket
import json
import base64
import time
import librosa
import numpy as np

# 配置
WAV_FILE = "Voice_service/output.wav"  # 你的音频文件路径
WS_URL = "ws://localhost:8056/v1/realtime"
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION = 0.04  # 40ms
CHUNK_SIZE = int(TARGET_SAMPLE_RATE * CHUNK_DURATION)

# 全局记录
final_transcript = []
start_time = None


def load_audio_chunks(path):
    audio, _ = librosa.load(path, sr=TARGET_SAMPLE_RATE)
    audio_int16 = (audio * 32767).astype(np.int16)
    chunks = [
        audio_int16[i:i + CHUNK_SIZE].tobytes()
        for i in range(0, len(audio_int16), CHUNK_SIZE)
    ]
    return chunks


def send_audio(ws, chunks):
    global start_time
    start_time = time.time()
    for chunk in chunks:
        audio_b64 = base64.b64encode(chunk).decode('utf-8')
        ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        time.sleep(CHUNK_DURATION)
    ws.send(json.dumps({"type": "end"}))


def on_message(ws, message):
    global final_transcript
    data = json.loads(message)
    if data.get("type") == "conversation.item.input_audio_transcription.completed":
        final_transcript.append(data["transcript"])


def on_open(ws):
    chunks = load_audio_chunks(WAV_FILE)
    send_audio(ws, chunks)


def on_close(ws, *args):
    if start_time:
        elapsed = time.time() - start_time
        print(f"🕒 识别耗时: {elapsed:.2f} 秒")
    print(f"📝 识别文本: {''.join(final_transcript)}")


if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close
    )
    ws.run_forever()
