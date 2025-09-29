import asyncio
import base64
import json
import wave
import websockets
import time

# --- 配置 ---
WEBSOCKET_URL = "ws://127.0.0.1:8056/v1/realtime"
WAV_FILE_PATH = "Voice_service/output.wav"  # 替换为你本地的 WAV 文件路径
CHUNK_DURATION_S = 0.1  # 每 100ms 一段音频
AUDIO_SAMPLE_RATE = 24000  # 要与服务端配置一致

# --- 音频读取 ---
def load_wav_chunks(file_path, chunk_duration, sample_rate):
    """读取wav文件并按时间切分为chunks（单位：秒）"""
    with wave.open(file_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "WAV必须是单声道"
        assert wf.getframerate() == sample_rate, f"WAV采样率必须为 {sample_rate}Hz"
        assert wf.getsampwidth() == 2, "WAV必须为16位"

        frames_per_chunk = int(chunk_duration * sample_rate)
        while True:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break
            yield frames

# --- 异步任务 ---
async def receive_task(ws):
    """接收服务端返回的识别结果"""
    try:
        async for message in ws:
            response = json.loads(message)
            print(f"[RECV] << {json.dumps(response, ensure_ascii=False)}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"连接关闭: {e.code}, {e.reason}")
    except Exception as e:
        print(f"接收时发生错误: {e}")

async def send_task(ws):
    """发送配置和WAV音频数据"""
    # 1. 发送初始配置信息
    config_message = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gummy-realtime-v1",
                "language": "zh",
            },
            "turn_detection": {"type": "server_vad"},
        },
    }
    await ws.send(json.dumps(config_message))
    print(f"[SEND] >> 初始配置已发送")
    await asyncio.sleep(0.1)

    # 2. 发送WAV音频数据（每段100ms）
    for chunk in load_wav_chunks(WAV_FILE_PATH, CHUNK_DURATION_S, AUDIO_SAMPLE_RATE):
        audio_b64 = base64.b64encode(chunk).decode("utf-8")
        audio_message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }
        await ws.send(json.dumps(audio_message))
        await asyncio.sleep(CHUNK_DURATION_S)

    print("--- WAV音频数据已全部发送 ---")

async def main():
    print(f"正在连接到: {WEBSOCKET_URL}")
    try:
        async with websockets.connect(WEBSOCKET_URL) as ws:
            await asyncio.gather(send_task(ws), receive_task(ws))
    except Exception as e:
        print(f"无法连接服务器: {e}")

if __name__ == "__main__":
    asyncio.run(main())
