from orpheus_tts import OrpheusModel
import wave
import time

model = OrpheusModel(model_name ="/data2/qinxb/model/orpheus-zh-pretrain")
# model = OrpheusModel(model_name="/data2/qinxb/model/orpheus-3b-0.1-ft")
prompt = '''
hello, how are you today, could you tell me some jokes?
'''
start_time = time.monotonic()
syn_tokens = model.generate_speech(
   prompt=prompt,
   voice=None,
   max_tokens=1000000,
   )

with wave.open("en.wav", "wb") as wf:
   wf.setnchannels(1)
   wf.setsampwidth(2)
   wf.setframerate(28000)

   total_frames = 0
   chunk_counter = 0
   for audio_chunk in syn_tokens: # output streaming
      chunk_counter += 1
      frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
      total_frames += frame_count
      wf.writeframes(audio_chunk)
   duration = total_frames / wf.getframerate()

   end_time = time.monotonic()
   print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")