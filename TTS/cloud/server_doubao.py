from flask import Flask, request, Response  
import requests  
import json  
import base64  
import os  
from threading import Thread  
import time  
import re

app = Flask(__name__)  
  
# 字节跳动TTS配置  
BYTEDANCE_CONFIG = {  
    "appID": "",  # 填入您的appID  
    "accessKey": "",  # 填入您的accessKey    
    "resourceID": "",  # 填入您的resourceID  
    "url": "https://openspeech.bytedance.com/api/v3/tts/unidirectional"  
}

def bytedance_tts_stream(text, speaker="zh_female_wanqudashu_moon_bigtts", audio_format="mp3", sample_rate=24000, emotion="happy", speed=0):  
    """字节跳动TTS流式合成"""  
    headers = {  
        "X-Api-App-Id": BYTEDANCE_CONFIG["appID"],  
        "X-Api-Access-Key": BYTEDANCE_CONFIG["accessKey"],  
        "X-Api-Resource-Id": BYTEDANCE_CONFIG["resourceID"],  
        "X-Api-App-Key": "aGjiRDfUWi",  
        "Content-Type": "application/json",  
        "Connection": "keep-alive"  
    }  
  
    additions = {  
        "disable_markdown_filter": True,  
        "enable_language_detector": True,  
        "enable_latex_tn": True,  
        "disable_default_bit_rate": True,  
        "max_length_to_filter_parenthesis": 0,  
        "cache_config": {  
            "text_type": 1,  
            "use_cache": True  
        }  
    }  
  
    payload = {
        "user": {"uid": "12345"},  
        "req_params": {  
            "text": text,  
            "speaker": speaker,  
            "additions": json.dumps(additions),  
            "audio_params": {  
                "format": audio_format,  
                "sample_rate": sample_rate,
                "emotion": emotion,
                "speech_rate": speed
            }  
        }  
    }  
  
    session = requests.Session()  
    try:  
        response = session.post(BYTEDANCE_CONFIG["url"], headers=headers, json=payload, stream=True)  
        
        for chunk in response.iter_lines(decode_unicode=True):  
            if not chunk:  
                continue  
              
            data = json.loads(chunk)  
            if data.get("code", 0) == 0 and "data" in data and data["data"]:  
                chunk_audio = base64.b64decode(data["data"])  
                yield chunk_audio  
            elif data.get("code", 0) == 20000000:  
                break  
            elif data.get("code", 0) > 0:  
                raise Exception(f"字节跳动TTS错误: {data}")  
        response.close()  
        session.close() 
                  
    except Exception as e:  
        raise Exception(f"字节跳动TTS请求失败: {e}")  
    
def clean_text(text):
    # 匹配四种模式并替换为空字符串
    pattern = r'(?:\[E[:：]\s*([a-zA-Z]+)\]|【E[:：]\s*([a-zA-Z]+)】)'
    return re.sub(pattern, '', text)

emotion = "neutral"

@app.route('/v1/audio/speech', methods=['POST'])  
def create_speech():  
    global emotion
    """OpenAI TTS API兼容端点"""  
    try:  
        data = request.get_json()  
        print(data)
        # 提取OpenAI标准参数  
        input_text = data.get('input', '')  
        model = data.get('model', 'tts-1')   
        voice = data.get('voice', 'zh_female_shuangkuaisisi_emo_v2_mars_bigtts')  
        speed = data.get('speed', 1.0)  
        response_format = data.get('response_format', 'mp3')  
        
        if not input_text:  
            return {"error": {"message": "input text is required"}}, 400  
        
        # 处理情感标记
        # pattern = r'\[E:([a-zA-Z]+)\]'
        pattern = r'(?:\[E[:：]\s*([a-zA-Z]+)\]|【E[:：]\s*([a-zA-Z]+)】)'

        match = re.search(pattern, input_text)  # 使用search只找第一个匹配
        if match:
            emotion = match.group(1) 

        input_text = clean_text(input_text)

        print("================EMOTION================")
        print(emotion)
        print("================TEXT==================")
        print(input_text)
        speed = 100*speed - 100

        bytedance_speaker = voice   
          
        def generate_audio():  
            try:  
                for chunk in bytedance_tts_stream(  
                    text=input_text,  
                    speaker=bytedance_speaker,  
                    audio_format=response_format,  
                    sample_rate=24000,
                    emotion=emotion,
                    speed=speed
                ):  
                    print(chunk[:20])
                    yield chunk  
            except Exception as e:  
                print(f"音频生成错误: {e}")  
                # 返回空数据表示错误  
                yield b''  
          
        return Response(  
            generate_audio(),  
            mimetype=f'audio/{response_format}',  
            headers={'Content-Type': f'audio/{response_format}'}  
        )  
          
    except Exception as e:  
        return {"error": {"message": str(e)}}, 500  
  
if __name__ == '__main__':  
    # 检查配置        
    app.run(host='0.0.0.0', port=8058, debug=False)