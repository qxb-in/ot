import asyncio  
from livekit.agents import APIConnectOptions  
  
async def simple_tts_test():  
    from livekit.plugins import openai  
      
    # tts = openai.TTS(  
    #     base_url="http://172.70.10.53:8059",
    #     api_key="sk-f6194dd95bcc47a199e9346a1fad1037",  
    #     model="cosyvoice-v1",  
    #     voice="longxiaochun",  
    #     speed=1.0,  
    #     response_format="wav"  
    # )
    tts = openai.TTS(  
        base_url="http://172.70.10.53:8058/v1",
        api_key="aa",  
        model="tts-1",  
        voice="zh_female_shuangkuaisisi_emo_v2_mars_bigtts",   
        response_format="wav",
    )
    
    test_text = "你好，我是联想智能客服助理，我想带你来一场智能之旅。你能不能给我一个测试，我想要看看这个到底能不能用"  
      
    try:  
        # 使用较短的超时时间进行测试  
        conn_options = APIConnectOptions(max_retry=1, timeout=10.0)  
          
        # async with tts.synthesize(test_text, conn_options=conn_options) as stream:  
        #     audio_frame = await stream.collect()  
        #     print(audio_frame)
        #     print(f"✅ TTS测试成功!")  
        #     print(f"   音频时长: {audio_frame.duration:.2f}秒")  
        #     print(f"   音频大小: {len(audio_frame.data)} bytes")  
        #     return True  
        async with tts.synthesize(test_text, conn_options=conn_options) as stream:  
            total_duration = 0.0  
            frame_count = 0  
            async for audio_event in stream:  
                frame_count += 1  
                total_duration += audio_event.frame.duration  
                print(f"音频段 {frame_count}: {audio_event.frame.duration:.3f}秒")  
            
            print(f"总共收到 {frame_count} 个音频段，总时长: {total_duration:.2f}秒")
            
        return True
    except Exception as e:  
        print(f"❌ TTS测试失败: {e}")  
        return False  
    finally:  
        await tts.aclose()  
  
# 运行测试  
success = asyncio.run(simple_tts_test())  
if success:  
    print("TTS模块工作正常!")  
else:  
    print("TTS模块存在问题，请检查服务配置。")


