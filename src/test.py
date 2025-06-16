import requests
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


def get_llm_response(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer GROQ_API_KEY",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    print("Raw response:", response.status_code, response.text)  # 👈 Add this line

    response.raise_for_status()  # raises an error if status code is 4xx/5xx
    return response.json()['choices'][0]['message']['content']


def synthesize_speech(text: str):
    # Load models (slow the first time)
    fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch")
    hifigan = HifiGanModel.from_pretrained("tts_hifigan")

    # Generate audio
    spectrogram = fastpitch.parse(text)
    spectrogram = fastpitch.generate_spectrogram(tokens=spectrogram)
    audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)

    # Save to file
    sf.write("output.wav", audio.to('cpu').detach().numpy()[0], 22050)

print(get_llm_response("What is Industry 5.0?"))
text = get_llm_response("Tell me a fun fact about robotics.")
synthesize_speech(text)