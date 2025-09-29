import asyncio  
import base64  
import hashlib  
import hmac  
import json  
import logging  
import ssl  
from datetime import datetime  
from time import mktime  
from urllib.parse import urlencode  
from wsgiref.handlers import format_date_time  
import io
import wave

import websockets  
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel  
  
# 配置日志  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
app = FastAPI(title="讯飞TTS代理服务", version="1.0.0")  
  
# 添加CORS中间件  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  
  
# 讯飞TTS配置  
XUNFEI_CONFIG = {  
    'APPID': '',  
    'API_KEY': '',  
    'API_SECRET': '',  
    'BASE_URL': 'wss://tts-api.xfyun.cn/v2/tts'  
}  
  
# 声音映射 - 将OpenAI声音映射到讯飞声音  
VOICE_MAPPING = {  
    'alloy': 'x4_yezi',  
    'echo': 'x4_lingfeng',   
    'fable': 'x4_xiaomo',  
    'onyx': 'x4_yezi',  
    'nova': 'x4_lingfeng',  
    'shimmer': 'x4_xiaomo'  
}  
  
# Pydantic模型定义  
class TTSRequest(BaseModel):  
    input: str  
    model: str = "tts-1"  
    voice: str = "alloy"  
    response_format: str = "mp3"  
    speed: float = 1.0  
  
class HealthResponse(BaseModel):  
    status: str  
    service: str  
  
class ModelInfo(BaseModel):  
    id: str  
    object: str  
    created: int  
    owned_by: str  
  
class ModelsResponse(BaseModel):  
    object: str  
    data: list[ModelInfo]  
  
class XunfeiTTSClient:  
    def __init__(self, appid: str, api_key: str, api_secret: str):  
        self.appid = appid  
        self.api_key = api_key  
        self.api_secret = api_secret  
          
    def create_auth_url(self) -> str:  
        """创建认证URL"""  
        url = XUNFEI_CONFIG['BASE_URL']  
          
        # 生成RFC1123格式的时间戳  
        now = datetime.now()  
        date = format_date_time(mktime(now.timetuple()))  
          
        # 拼接字符串  
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"  
        signature_origin += "date: " + date + "\n"  
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"  
          
        # 进行hmac-sha256加密  
        signature_sha = hmac.new(  
            self.api_secret.encode('utf-8'),   
            signature_origin.encode('utf-8'),  
            digestmod=hashlib.sha256  
        ).digest()  
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')  
          
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'  
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')  
          
        # 将请求的鉴权参数组合为字典  
        v = {  
            "authorization": authorization,  
            "date": date,  
            "host": "ws-api.xfyun.cn"  
        }  
          
        # 拼接鉴权参数，生成url  
        url = url + '?' + urlencode(v)  
        return url  
      
  
# 全局TTS客户端  
tts_client = XunfeiTTSClient(  
    XUNFEI_CONFIG['APPID'],  
    XUNFEI_CONFIG['API_KEY'],   
    XUNFEI_CONFIG['API_SECRET']  
)  
  
@app.post("/v1/audio/speech")
async def speech_synthesis(request: TTSRequest):
    """OpenAI兼容的TTS接口（流式传输）"""
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # 映射声音
    xunfei_voice = VOICE_MAPPING.get(request.voice, 'x4_yezi')
    
    logger.info(f"TTS请求 - 文本: {request.input[:50]}..., 声音: {request.voice} -> {xunfei_voice}")
    
    # 设置媒体类型
    mime_type = f'audio/{request.response_format}' if request.response_format != 'pcm' else 'audio/wav'
    
    # 创建流式响应
    async def generate_audio():
        """生成音频块的异步生成器"""
        # 创建认证URL
        url = tts_client.create_auth_url()
        
        # 准备请求数据
        request_data = {
            "common": {"app_id": tts_client.appid},
            "business": {
                "aue": "raw",
                "auf": "audio/L16;rate=16000", 
                "vcn": xunfei_voice,
                "tte": "utf8"
            },
            "data": {
                "status": 2,
                "text": str(base64.b64encode(request.input.encode('utf-8')), "UTF8")
            }
        }
        
        # 创建SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            # 收集所有PCM数据  
            pcm_data = bytearray() 

            async with websockets.connect(url, ssl=ssl_context) as websocket:
                # 发送请求
                await websocket.send(json.dumps(request_data))
                logger.info(f"发送TTS请求: {request.input[:50]}...")
                
                # 接收并流式传输响应
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    code = data.get("code", 0)
                    if code != 0:
                        error_msg = data.get("message", "Unknown error")
                        logger.error(f"讯飞TTS错误: {error_msg} (code: {code})")
                        raise HTTPException(status_code=500, detail=f"讯飞TTS错误: {error_msg}")
                    
                    # 获取音频数据
                    audio_data = data.get("data", {}).get("audio", "")

                    if audio_data:
                        audio_chunk = base64.b64decode(audio_data)
                        # wav_buffer = io.BytesIO()  
                        # with wave.open(wav_buffer, 'wb') as wav_file:  
                        #     wav_file.setnchannels(1)  # 单声道  
                        #     wav_file.setsampwidth(2)  # 16位  
                        #     wav_file.setframerate(16000)  # 16kHz  
                        #     wav_file.writeframes(audio_chunk)  
                        
                        # yield wav_buffer.getvalue() 
                        pcm_data.extend(audio_chunk)
                    # 转换为WAV格式（如果需要MP3，需要额外的编码库）  
                    
                    # 检查是否为最后一帧
                    status = data.get("data", {}).get("status", 0)
                    if status == 2:  # 最后一帧
                        logger.info("TTS合成完成")
                        break

            if request.response_format == "wav":  
                wav_buffer = io.BytesIO()  
                with wave.open(wav_buffer, 'wb') as wav_file:  
                    wav_file.setnchannels(1)  # 单声道  
                    wav_file.setsampwidth(2)  # 16位  
                    wav_file.setframerate(16000)  # 16kHz  
                    wav_file.writeframes(pcm_data)  
                
                yield wav_buffer.getvalue()  
            else:  
                # 对于其他格式，需要相应的编码处理  
                yield bytes(pcm_data)      
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"WebSocket连接提前关闭: {str(e)}")
        except Exception as e:
            logger.error(f"TTS合成错误: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # 返回流式响应
    return StreamingResponse(
        content=generate_audio(),
        media_type=mime_type,
        headers={
            'Content-Type': mime_type,
            'Transfer-Encoding': 'chunked'
        }
    )
  
@app.on_event("startup")  
async def startup_event():  
    """启动时检查配置"""  
    if not all([XUNFEI_CONFIG['APPID'], XUNFEI_CONFIG['API_KEY'], XUNFEI_CONFIG['API_SECRET']]):  
        logger.error("请设置讯飞TTS配置环境变量: XUNFEI_APPID, XUNFEI_API_KEY, XUNFEI_API_SECRET")  
        raise RuntimeError("Missing required environment variables")  
      
    logger.info("讯飞TTS代理服务启动完成")  
  
if __name__ == '__main__':  
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8055, log_level="info")