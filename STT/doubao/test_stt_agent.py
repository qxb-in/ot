import asyncio
import numpy as np
import resampy
import time
from livekit import rtc
from livekit.plugins import openai
from livekit.agents import stt

# --- 配置 ---
# 您的 Flask STT 代理服务的 URL
STT_PROXY_URL = "http://127.0.0.1:8056"

# 模拟音频生成设置
DUMMY_AUDIO_DURATION_S = 10  # 模拟 10 秒的音频
DUMMY_SAMPLE_RATE = 16000  # 以 16kHz 生成音频
DUMMY_FREQUENCY_HZ = 440  # 生成一个 A4 音调 (440Hz)
DUMMY_CHUNK_S = 0.1  # 模拟每次处理 100ms 的数据块

# LiveKit Agent 的 STT 插件期望的音频参数
AGENT_SAMPLE_RATE = 24000
AGENT_CHANNELS = 1
AGENT_FRAME_DURATION_MS = 20  # 20ms 的音频帧

# --- 音频生成器 ---
def generate_dummy_audio_chunk(start_time, chunk_duration, sample_rate, frequency):
    """生成一段正弦波音频块。"""
    t = np.linspace(
        start_time,
        start_time + chunk_duration,
        int(chunk_duration * sample_rate),
        endpoint=False,
    )
    amplitude = np.iinfo(np.int16).max * 0.5  # 50% 音量
    data = amplitude * np.sin(2 * np.pi * frequency * t)
    return data.astype(np.int16)


# --- 主要测试逻辑 ---
async def listen_for_transcripts(event_queue: asyncio.Queue[stt.SpeechEvent]):
    """从 Agent 监听并打印 STT 事件。"""
    print("转写结果监听已启动...")
    while True:
        try:
            event = await event_queue.get()
            # 由于输入的是合成音，我们主要关心是否能收到事件，内容可能为空
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = event.alternatives[0].text
                print(f"\n  >> 最终结果: '{transcript}'")
            elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                transcript = event.alternatives[0].text
                print(f"  >> 临时结果: '{transcript}'", end="\r")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"监听发生错误: {e}")
            break


async def main():
    """运行 STT 测试的主函数。"""
    print("--- LiveKit Agent STT 兼容性测试 (无麦克风模式) ---")
    print(f"正在连接到 STT 代理服务: {STT_PROXY_URL}")
    print(f"将模拟发送 {DUMMY_AUDIO_DURATION_S} 秒的合成音频...")

    # 1. 初始化 OpenAI STT 插件，并将其指向我们的代理服务
    stt_plugin = openai.STT(
        use_realtime=True,
        base_url=STT_PROXY_URL,
        api_key="dummy-key",  # 客户端要求提供此参数，但我们的代理服务不会使用它
    )

    # 2. 从插件获取一个语音流
    stream = stt_plugin.stream()

    # 3. 启动一个后台任务来监听转写结果
    listener_task = asyncio.create_task(listen_for_transcripts(stream.event_ch))

    try:
        total_chunks = int(DUMMY_AUDIO_DURATION_S / DUMMY_CHUNK_S)
        for i in range(total_chunks):
            # a. 生成一段模拟音频数据
            start_time = i * DUMMY_CHUNK_S
            dummy_data_chunk = generate_dummy_audio_chunk(
                start_time, DUMMY_CHUNK_S, DUMMY_SAMPLE_RATE, DUMMY_FREQUENCY_HZ
            )

            # b. 将数据从 16kHz 重采样到 Agent 需要的 24kHz
            agent_s16 = resampy.resample(
                dummy_data_chunk,
                DUMMY_SAMPLE_RATE,
                AGENT_SAMPLE_RATE,
                filter="kaiser_fast",
            ).astype(np.int16)

            # c. 将重采样后的音频分割成 20ms 的帧并发送给 Agent
            samples_per_frame = (AGENT_SAMPLE_RATE * AGENT_FRAME_DURATION_MS) // 1000
            offset = 0
            while offset + samples_per_frame <= len(agent_s16):
                frame_data = agent_s16[offset : offset + samples_per_frame]
                frame = rtc.AudioFrame(
                    data=frame_data.tobytes(),
                    sample_rate=AGENT_SAMPLE_RATE,
                    num_channels=AGENT_CHANNELS,
                    samples_per_frame=samples_per_frame,
                )
                await stream.input_ch.put(frame)
                offset += samples_per_frame
            
            print(f"  ...正在发送音频数据块 {i + 1}/{total_chunks}", end="\r")

            # d. 暂停一下，模拟实时音频流
            await asyncio.sleep(DUMMY_CHUNK_S)

    except KeyboardInterrupt:
        print("\n\n测试被中断，正在停止...")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 5. 清理资源
        print("\n\n音频发送完毕，正在关闭并清理资源...")
        await stream.aclose()
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass  # 这是预期的取消操作

        print("测试完成。")


if __name__ == "__main__":
    # 在执行此脚本前，请确保您的 Flask 代理服务 (app.py) 正在运行。
    asyncio.run(main()) 