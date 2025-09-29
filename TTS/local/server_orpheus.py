from flask import Flask, Response, request, render_template
import struct
from orpheus_tts import OrpheusModel
import os
from datetime import datetime

def generate_wav_filename(name):
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{name}_{now}.wav"


app = Flask(__name__)

engine_en = OrpheusModel(model_name="model/orpheus-zh-ft")
sample_rate_en=24000

engine_zh = OrpheusModel(model_name="model/orpheus-zh-pretrain")
sample_rate_zh=32000

def create_wav_header(sample_rate, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', '')
    
    lang_param = request.args.get('lang', 'zh')
    
    if lang_param == "en":
        voice = request.args.get('voice', 'zoe')
        if voice == "默认":
            voice = None
        sample_rate = sample_rate_en
    else:
        voice = request.args.get('voice', '白芷')
        if voice == "默认":
            voice = None
        sample_rate = sample_rate_zh

    name = prompt + '_' + voice
    filename = generate_wav_filename(name)
    filepath = os.path.join("./output", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


    def generate_audio_stream():
        with open(filepath, "wb") as f:
            # 写入WAV头（先写一个假的 data_size 为0，后面再回填）
            f.write(create_wav_header(sample_rate))

            total_audio_data = b''

            yield create_wav_header(sample_rate)

            if lang_param == "en":
                syn_tokens = engine_en.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    repetition_penalty=1.1,
                    stop_token_ids=[128258],
                    max_tokens=2000,
                    temperature=0.4,
                    top_p=0.9
                )
                for chunk in syn_tokens:
                    yield chunk
                    total_audio_data += chunk
                    f.write(chunk)
                
            elif lang_param == "zh":
                syn_tokens = engine_zh.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    repetition_penalty=1.1,
                    stop_token_ids=[128258],
                    max_tokens=2000,
                    temperature=0.4,
                    top_p=0.9
                )
                for chunk in syn_tokens:
                    yield chunk
                    total_audio_data += chunk
                    f.write(chunk)

            # 回填 WAV 文件大小（RIFF chunk size 和 data chunk size）
            data_size = len(total_audio_data)
            riff_size = 36 + data_size
            with open(filepath, "r+b") as fw:
                fw.seek(4)
                fw.write(struct.pack('<I', riff_size))
                fw.seek(40)
                fw.write(struct.pack('<I', data_size))

            print(f"[保存成功] 音频文件保存在: {filepath}")

    return Response(generate_audio_stream(), mimetype='audio/wav')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, threaded=True)