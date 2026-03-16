import os
import uuid
from pathlib import Path
import wave
import streamlit as st
import numpy as np
import subprocess

from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# =====================================
# Paths
# =====================================
UPLOAD_DIR = Path("data/uploads")
VOICE_PATH = Path("voices/voice.onnx")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# start Piper once
piper_process = subprocess.Popen(
    [
        "piper",
        "--model", "voices/voice.onnx",
        "--output_raw"
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
# =====================================
# Load Whisper Model (Speech → Text)
# =====================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        "base",
        device="cpu",         
        compute_type="int8",
        num_workers=4
    )


whisper_model = load_whisper_model()


# =====================================
# Load Piper Voice (Text → Speech)
# =====================================
@st.cache_resource
def load_piper_voice():
    return PiperVoice.load(str(VOICE_PATH))


piper_voice = load_piper_voice()


# ==================================
# Speech → Text
# ==================================

def speech_to_text_whisper(audio_path):

    segments, _ = whisper_model.transcribe(
        audio_path,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


# ==================================
# Text → Speech
# ==================================
def text_to_speech(text):

    file_path = f"data/uploads/tts_{uuid.uuid4().hex}.wav"

    command = [
        "piper",
        "--model", "voices/voice.onnx",
        "--output_file", file_path
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        text=True
    )

    process.communicate(text)

    return file_path