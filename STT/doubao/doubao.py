import asyncio
import websockets
import uuid
import json
import gzip
from typing import AsyncGenerator

appid = ""
token = ""
cluster = ""
api_url = f"wss://openspeech.bytedance.com/api/v1/tts/ws_binary"

default_header = bytearray(b'\x11\x10\x11\x00')

async def stream_tts(text: str, voice: str = "zh_female_shuangkuaisisi_moon_bigtts",
                     speed: float = 1.0, chunk_size: int = 2048) -> AsyncGenerator[bytes, None]:

    """
    通过 WebSocket 与字节跳动 openspeech 平台建立连接，发送待合成文本请求，并以流式方式实时接收语音数据（mp3 或 wav 格式）。
    每接收到一个音频 chunk 时会立即 yield 输出。

    参数：
        text (str): 待合成的文本内容，支持中文及部分英文。
        voice (str): 发音人参数，具体发音人见官方文档，默认为 "zh_female_shuangkuaisisi_moon_bigtts"。
        speed (float): 语速控制，默认 1.0 表示正常速度，可根据需要加快或放慢。
        chunk_size (int): 输出音频 chunk 的最小字节数，积累到该大小后就 yield，减小则延迟更低但可能更碎片化。

    返回：
        AsyncGenerator[bytes, None]: 异步生成器，逐段 yield 语音二进制数据（用于播放或保存）。
    """

    request_template = {
        "app": {
            "appid": appid,
            "token": token,
            "cluster": cluster
        },
        "user": {
            "uid": ""
        },
        "audio": {
            "voice": voice,
            "encoding": "mp3",  # 使用 mp3 减少体积（低延迟传输）
            "speed_ratio": speed,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "submit"
        }
    }

    payload = gzip.compress(json.dumps(request_template).encode("utf-8"))
    header_bytes = bytearray(default_header)
    header_bytes.extend(len(payload).to_bytes(4, 'big'))
    header_bytes.extend(payload)

    headers = {"Authorization": f"Bearer; {token}"}
    buffer = bytearray()

    async with websockets.connect(api_url, additional_headers=headers, ping_interval=None) as ws:
        await ws.send(header_bytes)
        while True:
            try:
                res = await ws.recv()
                done = await parse_chunk_response(res, buffer)

                # 如果缓冲区已积累 chunk_size ，尽快 yield
                while len(buffer) >= chunk_size:
                    yield bytes(buffer[:chunk_size])
                    del buffer[:chunk_size]
                if done:
                    if buffer:
                        # 返回最后一个 chunk
                        yield bytes(buffer)
                    break
            except websockets.ConnectionClosed as e:
                print(f"连接关闭：{e}")
                break

async def parse_chunk_response(res: bytes, buffer: bytearray) -> bool:

    """
    作用，识别消息类型，抽取并拼接音频数据到 buffer
    根据 sequence_number 判断合成是否结束
    捕获并抛出语音合成相关错误
    """
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    message_compression = res[2] & 0x0f
    header_size = res[0] & 0x0f
    payload = res[header_size * 4:]

    if message_type == 0xb:
        if message_type_specific_flags == 0:
            return False
        else:
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            audio = payload[8:]
            buffer.extend(audio)
            return sequence_number < 0
    elif message_type == 0xf:
        code = int.from_bytes(payload[:4], "big", signed=False)
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        raise RuntimeError(f"语音合成错误：{error_msg.decode('utf-8')}")
    return False


async def main():

    text = "您好，我们是客服中心，本次给您来电呢，是想和您分享一下最新产品和解决方案，请问您有兴趣了解一下吗？"
    voice="zh_male_shenyeboke_moon_bigtts"

    speed=1.2
    chunk_size=4096

    # output_path = "output.mp3"
    # with open(output_path, "wb") as f:  # 以二进制写入方式打开文件
    #     async for chunk in stream_tts(text, voice, speed, chunk_size):
    #         print(f"收到 chunk 大小: {len(chunk)} 字节")
    #         f.write(chunk)  # 直接写入音频 chunk 到文件

    async for chunk in stream_tts(text, voice, speed, chunk_size):
        # 你可以写入文件，发送给播放器，或直接打印长度
        print(f"收到 chunk 大小: {len(chunk)} 字节")

if __name__ == "__main__":
    asyncio.run(main())
