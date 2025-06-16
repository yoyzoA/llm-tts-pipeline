# 🧠 LLM-to-TTS Voice Agent (WSL + NeMo + Groq API)

This project builds a simple voice assistant that:
- Sends prompts to a **hosted LLM** (via [Groq API](https://console.groq.com/))
- Converts responses to speech using **NVIDIA NeMo TTS**
- Outputs an audio file (`output.wav`)

It runs inside **WSL (Ubuntu)** on Windows for full Linux compatibility.

---

## ⚙️ Windows & WSL Setup

### 1. Enable WSL2 and Install Ubuntu
Open PowerShell (Admin) and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Then **restart your system** when prompted.

> 📝 This installs:
> - WSL2 backend
> - Ubuntu 22.04 (Linux)
> - Linux filesystem access via `\\wsl$`

---

### 2. Open WSL and Install Python

Launch **Ubuntu (WSL)** from the Start menu and run:

```bash
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and development tools
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential \
                    libsndfile1 ffmpeg wget git
```

---

## 🐍 Python Project Setup

### 3. Create a Folder and Virtual Environment

```bash
mkdir ~/llm_tts
cd ~/llm_tts
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install nemo_toolkit[tts] torch soundfile requests
```

---

## 🧠 Script: `test.py`

Create a file called `test.py` with the following content:

```python
import requests
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

def get_llm_response(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer YOUR_GROQ_API_KEY",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def synthesize_speech(text: str):
    fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch")
    hifigan = HifiGanModel.from_pretrained("tts_hifigan")
    spec = fastpitch.generate_spectrogram(tokens=fastpitch.parse(text))
    audio = hifigan.convert_spectrogram_to_audio(spec)
    sf.write("output.wav", audio[0].cpu().numpy(), 22050)

print(get_llm_response("What is Industry 5.0?"))
text = get_llm_response("Tell me a fun fact about robotics.")
synthesize_speech(text)
```

> ⚠️ Replace `YOUR_GROQ_API_KEY` with your actual key from https://console.groq.com

---

## ▶️ Run the Script

```bash
source venv/bin/activate
python test.py
```

This will:
- Print 2 LLM responses
- Create `output.wav` using NeMo TTS

---

## 🔊 Play the Audio

### Option 1: Inside WSL

```bash
ffplay output.wav
```

### Option 2: Copy to Windows

```bash
cp output.wav /mnt/c/Users/YOUR_NAME/Desktop/
```

Then play it from your Windows Desktop.

---

## 📁 Optional: Access WSL Folder in Windows

You can open your WSL project directly in Windows Explorer:

```
\\wsl$\Ubuntu-22.04\home\yorgo\llm_tts
```

Or open it in **VS Code** using the "Remote - WSL" extension.

---

## ✅ Summary Commands

```bash
# WSL
wsl --install -d Ubuntu-22.04

# Inside WSL
sudo apt update && sudo apt install python3.11 python3.11-venv ...
mkdir ~/llm_tts && cd ~/llm_tts
python3.11 -m venv venv && source venv/bin/activate
pip install nemo_toolkit[tts] torch soundfile requests
python test.py
```

---

## 💡 Future Ideas

- Add ASR (speech-to-text) using NeMo
- Build a voice chatbot using FastAPI or Streamlit
- Use OpenAI, Together, or Mistral APIs as LLM sources

---

## 🧠 Credits

Built with:
- [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [Groq LLM API](https://console.groq.com)
- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/)
