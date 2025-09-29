from flask import Flask, request, Response  
import dashscope  
from dashscope.audio.tts_v2 import *  
import json  
import time  
from threading import Thread  
from queue import Queue  
  
app = Flask(__name__)  

dashscope.api_key = ""  
  
class ChunkGenerator:  
    def __init__(self):  
        self.queue = Queue()  
        self.finished = False  
        self.error = None  
  
    def put(self, data):  
        self.queue.put(data)  
      
    def end(self):  
        self.finished = True  
      
    def set_error(self, message):  
        self.error = message  
        self.end()  
      
    def generate(self):  
        while not self.finished or not self.queue.empty():  
            if not self.queue.empty():  
                yield self.queue.get()  
            else:  
                time.sleep(0.01)  
          
        if self.error:  
            raise Exception(f"Speech synthesis failed: {self.error}")  
  
class Callback(ResultCallback):  
    def __init__(self, chunk_generator):  
        self.chunk_generator = chunk_generator  
      
    def on_open(self):  
        pass  
      
    def on_complete(self):  
        self.chunk_generator.end()  
      
    def on_error(self, message: str):  
        self.chunk_generator.set_error(message)  
      
    def on_close(self):  
        self.chunk_generator.end()  
      
    def on_data(self, data: bytes) -> None:  
        self.chunk_generator.put(data)  
  
@app.route('/audio/speech', methods=['POST'])  
def create_speech():  
    """OpenAI TTS API兼容端点"""  
    try:  
        data = request.get_json()  
        print(data)
        # 提取OpenAI标准参数  
        input_text = data.get('input', '')  
        api_key = data.get('api_key', '')
        model = data.get('model', 'cosyvoice-v1')  
        voice = data.get('voice', 'longxiaochun')  
        speed = data.get('speed', 1.0)  
        response_format = data.get('response_format', 'wav')  
          
        if not input_text:  
            return {"error": {"message": "input text is required"}}, 400  
          
        # 映射音频格式  
        audio_format_map = {  
            'wav': AudioFormat.WAV_24000HZ_MONO_16BIT,   
            'mp3': AudioFormat.MP3_24000HZ_MONO_256KBPS
        }  
          
        qwen_format = audio_format_map.get(response_format, AudioFormat.WAV_24000HZ_MONO_16BIT)  
          
        def generate_audio():  
            chunk_generator = ChunkGenerator()  
            callback = Callback(chunk_generator)  
              
            synthesizer = SpeechSynthesizer(  
                model=model,  
                voice=voice,  
                speech_rate=speed,  
                format=qwen_format,  
                callback=callback,  
            )  
              
            def run_synthesis():  
                try:  
                    synthesizer.streaming_call(input_text)  
                    synthesizer.streaming_complete()  
                except Exception as e:  
                    chunk_generator.set_error(str(e))  
              
            Thread(target=run_synthesis).start()  
              
            try:  
                for chunk in chunk_generator.generate():  
                    yield chunk  
            except Exception as e:  
                # 如果生成过程中出错，返回错误响应  
                print(e)
                yield b''  # 空响应表示错误  
          
        return Response(  
            generate_audio(),  
            mimetype=f'audio/{response_format}',  
            headers={  
                'Content-Type': f'audio/{response_format}',   
            }  
        )  
          
    except Exception as e:  
        return {"error": {"message": str(e)}}, 500  
  
@app.route('/v1/audio/speech', methods=['POST'])  
def create_speech_v1():  
    """支持/v1/audio/speech路径"""  
    return create_speech()  
  
if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=8059, debug=False)