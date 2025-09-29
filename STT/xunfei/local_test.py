import asyncio  
import aiohttp  
import os  
import time  
import wave  
import numpy as np  
from pathlib import Path  
  
from livekit.plugins import openai  
from livekit.agents.utils import AudioBuffer  
from livekit import rtc  
  
# 设置虚拟 API key  
os.environ["OPENAI_API_KEY"] = "key"  
  
# 测试配置  
TEST_WAV_FILE = "test/audio_samples/舞台_有背景音.wav"  
XUNFEI_SERVICE_URL = "http://:8056"  
  
async def load_wav_file(file_path: str) -> AudioBuffer:  
    """正确加载 WAV 文件"""  
    with wave.open(file_path, 'rb') as wav_file:  
        sample_rate = wav_file.getframerate()  
        num_channels = wav_file.getnchannels()  
        frames = wav_file.readframes(-1)  
          
        if wav_file.getsampwidth() == 2:  
            audio_data = np.frombuffer(frames, dtype=np.int16)  
        else:  
            raise ValueError("只支持 16-bit WAV 文件")  
          
        if num_channels == 2:  
            audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)  
          
        # 直接使用字节数据创建 AudioFrame  
        audio_frame = rtc.AudioFrame(  
            data=audio_data.tobytes(),  # 转换为字节  
            sample_rate=sample_rate,  
            num_channels=1,  
            samples_per_channel=len(audio_data)  
        )  
          
        return [audio_frame]  
  
async def test_livekit_stt():  
    """使用 LiveKit OpenAI STT 模块测试"""  
    print("=== LiveKit STT 测试 ===")  
      
    stt = openai.STT(  
        base_url=f"{XUNFEI_SERVICE_URL}/v1",  
        model="whisper-1",  
        api_key="dummy-key-for-custom-endpoint"  
    )  
      
    if not os.path.exists(TEST_WAV_FILE):  
        print(f"错误: 找不到音频文件 {TEST_WAV_FILE}")  
        return  
      
    try:  
        buffer = await load_wav_file(TEST_WAV_FILE)  
          
        start_time = time.time()  
        event = await stt.recognize(buffer=buffer)  
        end_time = time.time()  
          
        print(f"识别结果: {event.alternatives[0].text}")  
        print(f"识别耗时: {end_time - start_time:.2f}秒")  
        print(f"事件类型: {event.type}")  
          
    except Exception as e:  
        print(f"LiveKit STT 测试失败: {e}") 


async def test_direct_api():  
    """直接调用包装服务 API 测试"""  
    print("\n=== 直接 API 调用测试 ===")  
      
    if not os.path.exists(TEST_WAV_FILE):  
        print(f"错误: 找不到音频文件 {TEST_WAV_FILE}")  
        return  
      
    try:  
        async with aiohttp.ClientSession() as session:  
            # 准备表单数据  
            with open(TEST_WAV_FILE, 'rb') as f:  
                form_data = aiohttp.FormData()  
                form_data.add_field('file', f, filename='test.wav', content_type='audio/wav')  
                form_data.add_field('model', 'whisper-1')  
                form_data.add_field('response_format', 'json')  
                  
                start_time = time.time()  
                  
                # 发送请求到包装服务  
                async with session.post(  
                    f"{XUNFEI_SERVICE_URL}/v1/audio/transcriptions",  
                    data=form_data  
                ) as response:  
                    result = await response.json()  
                      
                end_time = time.time()  
                  
                print(f"识别结果: {result.get('text', '无结果')}")  
                print(f"识别耗时: {end_time - start_time:.2f}秒")  
                print(f"HTTP 状态码: {response.status}")  
                  
    except Exception as e:  
        print(f"直接 API 调用测试失败: {e}")  
  
  
  
async def main():  
    """主测试函数"""  
    print("讯飞 STT 包装服务测试")  
    print("=" * 50)  
      
    # 检查服务是否运行  
    try:  
        async with aiohttp.ClientSession() as session:  
            async with session.get(f"{XUNFEI_SERVICE_URL}/docs") as response:  
                if response.status == 200:  
                    print("✓ 讯飞包装服务正在运行")  
                else:  
                    print("✗ 讯飞包装服务未响应")  
                    return  
    except:  
        print("✗ 无法连接到讯飞包装服务，请确保服务已启动")  
        return  
      
    # 运行测试  
    # await test_livekit_stt()  
    await test_direct_api()  
      
    print("\n测试完成!")  
  
if __name__ == "__main__":  
    asyncio.run(main())