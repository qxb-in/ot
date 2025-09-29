import base64
import json
import logging
import os
import queue
import threading
from pathlib import Path

import numpy as np
import resampy
from flask import Flask, request
from flask_sock import Sock, ConnectionClosed

import dashscope
from dashscope.audio.asr import *
from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerRealtime, TranscriptionResult

# --- 配置 ---
# 日志配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dashscope API Key 配置
# 推荐您将API Key存储在环境变量 `DASHSCOPE_API_KEY` 中
# 如果环境变量未设置，请在此处填写您的 key
if "DASHSCOPE_API_KEY" not in os.environ:
    dashscope.api_key = ""  # 请替换为您的 API Key
    if not dashscope.api_key:
        logging.warning(
            "DASHSCOPE_API_KEY 环境变量未设置，并且代码中也未提供。请务必配置 API Key。"
        )

# 音频参数
# livekit-agent 的 OpenAI 插件默认使用 24kHz 采样率
SOURCE_SAMPLE_RATE = 16000
# Dashscope `paraformer-realtime-v1` 模型要求 16kHz 采样率
TARGET_SAMPLE_RATE = 16000
# Dashscope 模型名称
DASHSCOPE_MODEL = "gummy-realtime-v1"

# --- Flask 应用设置 ---
app = Flask(__name__)
sock = Sock(app)


# --- Dashscope 实时识别回调 ---
class DashscopeCallback(TranslationRecognizerCallback):
    """处理 Dashscope 识别事件并通过 WebSocket 发送结果。"""

    def __init__(self, ws_conn):
        super().__init__()
        self.ws_conn = ws_conn
        self.last_interim_text = ""
        self.last_final_text = ""

    def on_open(self) -> None:
        logging.info("Dashscope recognizer connection opened.")

    def on_close(self) -> None:
        logging.info("Dashscope recognizer connection closed.")

    def on_error(self, result: TranscriptionResult) -> None:
        logging.error(f"Dashscope recognizer error: {result}")
    
    def on_event(self, request_id, transcription_result, translation_result, usage):    
        if transcription_result is not None:    
            transcript = transcription_result.text    
            is_final = transcription_result.stash is None    
            
            if transcript:  
                if not is_final:  
                    # 中间结果 - 发送增量  
                    if transcript != self.last_interim_text:  
                        delta = transcript[len(self.last_interim_text):]  
                        if delta:  
                            response = {    
                                "type": "conversation.item.input_audio_transcription.delta",    
                                "delta": delta,    
                            }    
                            self.ws_conn.send(json.dumps(response))  
                            self.last_interim_text = transcript  
                            logging.info(f"Delta: '{delta}'")  
                else:  
                    # 最终结果 - 发送完成事件  
                    response = {    
                        "type": "conversation.item.input_audio_transcription.completed",    
                        "transcript": transcript,    
                    }    
                    self.ws_conn.send(json.dumps(response))    
                    self.last_final_text = transcript  
                    self.last_interim_text = ""  # 重置  
                    logging.info(f"Final transcript: '{transcript}'")


# --- WebSocket 路由 ---
@sock.route("/v1/realtime")
def realtime_stt(ws):
    """处理实时语音识别的 WebSocket 连接。"""
    logging.info("WebSocket connection established.")

    # 1. 接收并记录客户端的初始配置信息
    try:
        config_message = ws.receive(timeout=5)
        config = json.loads(config_message)
        print(config)
        logging.info(f"Received client configuration: {json.dumps(config, indent=2)}")
    except Exception as e:
        logging.warning(f"Did not receive client configuration or it was invalid: {e}")
        return  # 如果没有配置，则无法继续

    # 2. 初始化 Dashscope Recognizer
    callback = DashscopeCallback(ws)
    try:
        recognizer = TranslationRecognizerRealtime(
            model=DASHSCOPE_MODEL,
            format="pcm",
            sample_rate=TARGET_SAMPLE_RATE,
            transcription_enabled=True,  
            translation_enabled=False,  
            callback=callback,
            source_language="zh"
        )
        recognizer.start()
    except Exception as e:
        logging.error(f"Failed to start Dashscope recognizer: {e}")
        ws.close()
        return

    # 3. 循环接收音频数据
    try:
        while True:
            message = ws.receive()
            data = json.loads(message)
            # print(data.keys())

            if data.get("type") == "input_audio_buffer.append":
                audio_b64 = data.get("audio")
                if audio_b64:
                    # 解码
                    audio_bytes = base64.b64decode(audio_b64)

                    # 重采样: 24kHz -> 16kHz
                    audio_s16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    resampled_audio = resampy.resample(
                        audio_s16,
                        SOURCE_SAMPLE_RATE,
                        TARGET_SAMPLE_RATE,
                        filter="kaiser_fast",  # 快速模式，适用于实时场景
                    )
                    resampled_bytes = resampled_audio.astype(np.int16).tobytes()

                    # 发送到 Dashscope
                    recognizer.send_audio_frame(resampled_bytes)
            else:
                logging.warning(f"Received unknown message type: {data.get('type')}")

    except ConnectionClosed:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"An error occurred in the WebSocket loop: {e}")
    finally:
        # 4. 清理资源
        logging.info("Closing connection and stopping recognizer.")
        recognizer.stop()


# --- 主程序入口 ---
if __name__ == "__main__":
    
    host = "0.0.0.0"
    port = 8255
    
    print(f"Starting Dashscope STT proxy server on http://{host}:{port}")
    app.run(host=host, port=port)
