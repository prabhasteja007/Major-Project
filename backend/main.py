import assemblyai as aai
import sounddevice as sd
import subprocess
import os
from groq import Groq
from scipy.io.wavfile import write
from api import api_speech, grog_api

# Set your AssemblyAI API key
aai.settings.api_key = api_speech

# Function to record audio from the microphone
def record_audio(filename, duration=5, samplerate=44100, channels=1):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    print("Finished recording.")
    write(filename, samplerate, audio_data)

#function to convert text to speech
def text_to_speech(text):
    subprocess.run(['osascript', '-e', f'say "{text}"'])

text_to_speech("welcome")

# Record audio from the microphone
record_audio("recorded_audio.wav")

# Transcribe the recorded audio
transcriber = aai.Transcriber()
transcript = transcriber.transcribe("./recorded_audio.wav")
content = transcript.text
print(content)

client = Groq(
    api_key=grog_api,
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],
    model="llama3-70b-8192",
)
text = chat_completion.choices[0].message.content
text_to_speech(text)

print(text)