from flask import Flask, Response, request, render_template
import struct
from orpheus_tts import OrpheusModel

lang = "zh"

app = Flask(__name__)
if lang == "en":
    engine = OrpheusModel(model_name="/data2/qinxb/model/orpheus-3b-0.1-ft")
    sample_rate=24000

if lang == "zh":
    engine = OrpheusModel(model_name="/data2/qinxb/model/orpheus-zh-ft")
    # sample_rate=24000

    # engine = OrpheusModel(model_name="/data2/qinxb/model/orpheus-zh-pretrain")
    sample_rate=32000

def create_wav_header(sample_rate=sample_rate, bits_per_sample=16, channels=1):
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

@app.route('/')  
def index(): 
    content_zh = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>中文 Orpheus TTS 在线体验</title>
        </head>
        <body>
            <h1>中文 Orpheus TTS 在线体验</h1>
            <form id="promptForm">
                <label for="promptInput">文本输入</label><br>
                <textarea id="promptInput" rows="4" cols="50" placeholder="输入文本" required></textarea><br>
                <label for="voiceSelect">选择语音：</label>
                <select id="voiceSelect" name="voice">
                    <option value="白芷">白芷</option>
                    <option value="长乐">长乐</option>
                    <option value="默认">默认</option>
                </select><br><br>
                <button type="submit">生成</button>
            </form>
            <audio id="audioPlayer" controls autoplay></audio>
            <script>
                const base_url = `http://127.0.0.1:8090`;
                document.getElementById("promptForm").addEventListener("submit", function(event) {
                    event.preventDefault();
                    const prompt = document.getElementById("promptInput").value;
                    const voice = document.getElementById("voiceSelect").value;
                    const encodedPrompt = encodeURIComponent(prompt);
                    const audioUrl = `${base_url}/tts?prompt=${encodedPrompt}&voice=${voice}`;
                    
                    const audioPlayer = document.getElementById("audioPlayer");
                    audioPlayer.src = audioUrl;
                    audioPlayer.load();
                    audioPlayer.play().catch(err => console.error("Playback error:", err));
                });
            </script>
        </body>
        </html>
    """
    content_en = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>英文 Orpheus TTS 在线体验</title>
        </head>
        <body>
            <h1>英文 Orpheus TTS 在线体验</h1>
            <form id="promptForm">
                <label for="promptInput">文本输入</label><br>
                <textarea id="promptInput" rows="4" cols="50" placeholder="输入文本" required></textarea><br>
                <label for="voiceSelect">选择语音：</label>
                <select id="voiceSelect" name="voice">
                    <option value="zoe">zoe</option>
                    <option value="zac">zac</option>
                    <option value="jess">jess</option>
                    <option value="leo">leo</option>
                    <option value="mia">mia</option>
                    <option value="julia">julia</option>
                    <option value="leah">leah</option>
                </select><br><br>
                <button type="submit">生成</button>
            </form>
            <audio id="audioPlayer" controls autoplay></audio>
            <script>
                const base_url = `http://127.0.0.1:8090`;
                document.getElementById("promptForm").addEventListener("submit", function(event) {
                    event.preventDefault();
                    const prompt = document.getElementById("promptInput").value;
                    const voice = document.getElementById("voiceSelect").value;
                    const encodedPrompt = encodeURIComponent(prompt);
                    const audioUrl = `${base_url}/tts?prompt=${encodedPrompt}&voice=${voice}`;
                    
                    const audioPlayer = document.getElementById("audioPlayer");
                    audioPlayer.src = audioUrl;
                    audioPlayer.load();
                    audioPlayer.play().catch(err => console.error("Playback error:", err));
                });
            </script>
        </body>
        </html>
    """

    if lang == "en":
        return content_en
    elif lang == "zh":
        return content_zh
    
@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', '')
    if lang == "en":
        voice = request.args.get('voice', 'zoe')
    elif lang == "zh":
        voice = request.args.get('voice', '白芷')
        print("=======================================================", "\n", voice)
        if voice == "默认":
            voice = "tara"
    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
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

    return Response(generate_audio_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, threaded=True)
