from fastapi import FastAPI, File, UploadFile, Form, HTTPException  
from fastapi.responses import JSONResponse  
import asyncio  
import aiohttp  
import base64  
import hashlib  
import hmac  
import json  
import time  
from typing import Optional  
import logging

logger = logging.getLogger("xunfei")

app = FastAPI()  
  

def extract_transcript_from_xunfei_result(result: dict) -> str:  
    """从讯飞返回结果中提取转录文本"""  
    transcript_text = ""  
      
    if 'content' not in result or 'orderResult' not in result['content']:  
        return transcript_text  
      
    try:  
        # orderResult 是一个 JSON 字符串，需要先解析  
        order_result_str = result['content']['orderResult']  
        order_result = json.loads(order_result_str)  
          
        # 使用 lattice2 数据，因为它包含更完整的结构化数据  
        for lattice in order_result.get('lattice2', []):  
            json_1best = lattice.get('json_1best', {})  
            if isinstance(json_1best, dict):  
                st = json_1best.get('st', {})  
                for rt_item in st.get('rt', []):  
                    for ws_item in rt_item.get('ws', []):  
                        for cw_item in ws_item.get('cw', []):  
                            word = cw_item.get('w', '')  
                            if word:  # 过滤空字符串  
                                transcript_text += word  
          
        # 如果 lattice2 没有数据，尝试使用 lattice  
        if not transcript_text:  
            for lattice in order_result.get('lattice', []):  
                json_1best_str = lattice.get('json_1best', '{}')  
                # json_1best 在 lattice 中也是字符串，需要再次解析  
                if isinstance(json_1best_str, str):  
                    json_1best = json.loads(json_1best_str)  
                    st = json_1best.get('st', {})  
                    for rt_item in st.get('rt', []):  
                        for ws_item in rt_item.get('ws', []):  
                            for cw_item in ws_item.get('cw', []):  
                                word = cw_item.get('w', '')  
                                if word:  # 过滤空字符串  
                                    transcript_text += word  
                                      
    except json.JSONDecodeError as e:  
        logger.error(f"解析讯飞返回结果失败: {e}")  
    except Exception as e:  
        logger.error(f"提取转录文本时发生错误: {e}")  
      
    return transcript_text

class XunfeiSTTWrapper:  
    def __init__(self, appid: str, secret_key: str):  
        self.appid = appid  
        self.secret_key = secret_key  
        self.lfasr_host = 'https://raasr.xfyun.cn/v2/api'  
          
    def get_signa(self, ts: str) -> str:  
        m2 = hashlib.md5()  
        m2.update((self.appid + ts).encode('utf-8'))  
        md5 = m2.hexdigest()  
        md5 = bytes(md5, encoding='utf-8')  
        signa = hmac.new(self.secret_key.encode('utf-8'), md5, hashlib.sha1).digest()  
        signa = base64.b64encode(signa)  
        return str(signa, 'utf-8')  
      
    async def transcribe_audio(self, audio_data: bytes, filename: str) -> dict:  
        ts = str(int(time.time()))  
        signa = self.get_signa(ts)  
          
        # 上传音频  
        upload_params = {  
            'appId': self.appid,  
            'signa': signa,  
            'ts': ts,  
            'fileSize': len(audio_data),  
            'fileName': filename,  
            'duration': "200"  
        }  
          
        async with aiohttp.ClientSession() as session:  
            # 上传请求  
            upload_url = f"{self.lfasr_host}/upload"  
            async with session.post(  
                upload_url,  
                params=upload_params,  
                headers={"Content-type": "application/json"},  
                data=audio_data  
            ) as response:  
                upload_result = await response.json()  
                  
            if 'content' not in upload_result or 'orderId' not in upload_result['content']:  
                raise HTTPException(status_code=500, detail="Upload failed")  
                  
            order_id = upload_result['content']['orderId']  
              
            # 轮询结果  
            result_params = {  
                'appId': self.appid,  
                'signa': signa,  
                'ts': ts,  
                'orderId': order_id,  
                'resultType': "transfer,predict"  
            }  
              
            status = 3  
            while status == 3:  
                async with session.post(  
                    f"{self.lfasr_host}/getResult",  
                    params=result_params,  
                    headers={"Content-type": "application/json"}  
                ) as response:  
                    result = await response.json()  
                    status = result['content']['orderInfo']['status']  
                      
                    if status == 4:  # 完成  
                        return result  
                    elif status == -1:  # 失败  
                        return result
                        # raise HTTPException(status_code=500, detail="Transcription failed")  
                          
                await asyncio.sleep(2)  # 等待2秒后重试  
  
# 初始化讯飞服务  
xunfei_stt = XunfeiSTTWrapper(  
    appid="",  
    secret_key=""  
)  
  
@app.post("/v1/audio/transcriptions")  
async def transcribe(  
    file: UploadFile = File(...),  
    model: str = Form("whisper-1"),  
    language: Optional[str] = Form(None),  
    response_format: str = Form("json")  
):  
    try:  
        audio_data = await file.read()  
        result = await xunfei_stt.transcribe_audio(audio_data, file.filename)  
          
        # 使用修复后的文本提取函数  
        transcript_text = extract_transcript_from_xunfei_result(result)  
        print(transcript_text)
        return JSONResponse({  
            "text": transcript_text  
        })  
          
    except Exception as e:  
        logger.exception(f"转录失败: {e}")  
        raise HTTPException(status_code=500, detail=str(e))
  
if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8056)