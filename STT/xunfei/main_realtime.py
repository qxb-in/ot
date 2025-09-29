# -*- coding:utf-8 -*-
import hashlib
import hmac
import base64
import json
import time
import threading
import logging
import os
import queue

from flask import Flask
from flask_sock import Sock, ConnectionClosed
from websocket import create_connection, WebSocketConnectionClosedException
from urllib.parse import quote

app = Flask(__name__)
sock = Sock(app)

# 实时语音转写
# 配置
app_id = ""
api_key = ""

XF_BASE_URL = "ws://rtasr.xfyun.cn/v1/ws"

# 日志配置
logging.basicConfig(level=logging.INFO)

# 内部缓冲区队列大小限制
MAX_QUEUE_SIZE = 100

def extract_text_from_xunfei(data):    
    text_parts = []  
    if not isinstance(data, dict):  
        return ""  
    if 'cn' in data and isinstance(data['cn'], dict) and 'st' in data['cn'] and isinstance(data['cn']['st'], dict) and 'rt' in data['cn']['st']:    
        for rt in data['cn']['st']['rt']:    
            if isinstance(rt, dict) and 'ws' in rt:    
                for ws in rt['ws']:    
                    if isinstance(ws, dict) and 'cw' in ws:    
                        for cw in ws['cw']:    
                            if isinstance(cw, dict) and 'w' in cw:    
                                text_parts.append(cw['w'])    
    return ''.join(text_parts)

class XfAsrProxy:
    def __init__(self, client_ws):
        self.client_ws = client_ws
        self.queue = queue.Queue(MAX_QUEUE_SIZE)
        self.end_tag = "{\"end\": true}"
        self.ws = None
        self.recv_thread = None
        self.send_thread = None
        self.running = False

    def start(self):
        # 生成鉴权参数
        ts = str(int(time.time()))
        tt = (app_id + ts).encode('utf-8')
        baseString = hashlib.md5(tt).hexdigest().encode("utf-8")
        signa = base64.b64encode(hmac.new(api_key.encode("utf-8"), baseString, hashlib.sha1).digest()).decode("utf-8")
        ws_url = f"{XF_BASE_URL}?appid={app_id}&ts={ts}&signa={quote(signa)}"

        # 连接讯飞 WebSocket
        self.ws = create_connection(ws_url)
        self.running = True

        # 启动发送和接收线程
        self.recv_thread = threading.Thread(target=self.recv)
        self.send_thread = threading.Thread(target=self.send)
        self.recv_thread.start()
        self.send_thread.start()

    def put_audio(self, audio_bytes):
        try:
            self.queue.put(audio_bytes, timeout=2)
        except queue.Full:
            logging.warning("Audio queue is full, dropping audio frame")

    def send(self):
        try:
            while self.running:
                chunk = self.queue.get()
                if chunk is None:
                    break
                self.ws.send(chunk)
                time.sleep(0.04)  # 必须要控制速率
            self.ws.send(self.end_tag.encode('utf-8'))
            logging.info("Send end tag.")
        except WebSocketConnectionClosedException:
            logging.warning("Connection closed during send.")
        except Exception as e:
            logging.error(f"Send error: {e}")

    def recv(self):
        try:
            while self.running and self.ws.connected:
                result = self.ws.recv()
                if not result:
                    break
                result_dict = json.loads(result)
                if result_dict["action"] == "result":
                    # 发送识别结果到前端
                    data = result_dict.get("data", "")
                    if data:
                        msg = {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "delta": data
                        }
                        self.client_ws.send(json.dumps(msg))
                elif result_dict["action"] == "error":
                    logging.error("RTASR error: " + result)
                    self.client_ws.send(json.dumps({"type": "error", "message": result}))
                    break
        except WebSocketConnectionClosedException:
            logging.info("WebSocket closed during recv.")
        except Exception as e:
            logging.error(f"Receive error: {e}")
        finally:
            self.close()

    def stop(self):
        self.running = False
        self.queue.put(None)  # 标记结束发送线程
        if self.ws:
            self.ws.close()
        logging.info("Connection to Xunfei closed.")

    def close(self):
        self.stop()
        try:
            self.client_ws.close()
        except:
            pass


# WebSocket 路由
@sock.route("/v1/realtime")
def handle_realtime(ws):
    logging.info("WebSocket client connected.")
    try:
        # 接收初始配置（可选）
        config = ws.receive()
        logging.info(f"Client config: {config}")

        proxy = XfAsrProxy(ws)
        proxy.start()

        while True:
            message = ws.receive()
            if not message:
                break
            data = json.loads(message)
            if data.get("type") == "input_audio_buffer.append":
                audio_b64 = data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    proxy.put_audio(audio_bytes)
            elif data.get("type") == "end":
                logging.info("Received end from client.")
                break
            else:
                logging.warning(f"Unknown message type: {data.get('type')}")

    except ConnectionClosed:
        logging.info("WebSocket connection closed by client.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        if 'proxy' in locals():
            proxy.close()
        logging.info("Session closed.")


# 启动 Flask 应用
if __name__ == '__main__':
    host = "0.0.0.0"
    port = 8056
    print(f"🚀 Starting Xunfei ASR server on ws://{host}:{port}/v1/realtime")
    app.run(host=host, port=port)
