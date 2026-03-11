# AI RAG Voice Assistant

A fully local **AI voice assistant** built with **RAG (Retrieval-Augmented Generation)** using Llama3, speech recognition, and text-to-speech.

The assistant can take **voice input**, retrieve relevant knowledge from a vector database, generate an answer using Llama3, and speak the response automatically.

---

# Features

* Voice input using microphone
* Automatic speech transcription
* Retrieval-Augmented Generation (RAG)
* Hybrid document retrieval
* Neural reranking
* Local Llama3 inference
* Automatic text-to-speech playback
* Fully local (no external APIs)

---

# Technologies Used

| Component         | Technology        |
| ----------------- | ----------------- |
| Interface         | Streamlit         |
| LLM               | Llama3 via Ollama |
| Speech-to-Text    | Faster-Whisper    |
| Text-to-Speech    | Piper             |
| Vector Search     | FAISS             |
| Embeddings        | BGE Embeddings    |
| Keyword Retrieval | BM25              |
| Reranking         | BGE Reranker      |

---

# System Architecture

User Input (Voice)
↓
Speech Recognition (Whisper)
↓
Hybrid Retrieval
• Vector Search (FAISS)
• Keyword Search (BM25)
↓
Neural Reranking
↓
Context Selection
↓
Llama3 Response Generation
↓
Text-to-Speech (Piper)
↓
Automatic Audio Playback

---

# Project Structure

```
AI_RAG_Chatbot_Project
│
├── app.py
├── rag_utils.py
├── audio_utils.py
├── requirements.txt
│
├── data
│   └── uploads
│
└── voices
    ├── voice.onnx
    └── voice.onnx.json
```

---

# Installation

## 1. Clone the repository

```
git clone <repository-url>
cd AI_RAG_Chatbot_Project
```

---

## 2. Create virtual environment

```
python -m venv venv
```

Activate it.

Windows:

```
venv\Scripts\activate
```

---

## 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download from:

https://ollama.com

Then install the Llama3 model:

```
ollama pull llama3
```

Start the Ollama server:

```
ollama serve
```

---

# Running the Application

Start the Streamlit interface:

```
streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

---

# Using the Assistant

### Voice Questions

Record audio using the microphone.

The system will:

1. Transcribe speech
2. Retrieve relevant documents
3. Generate an answer
4. Speak the response automatically

---

# Performance

Typical response time on a local machine:

| Component          | Time    |
| ------------------ | ------- |
| Speech recognition | ~0.5 s  |
| RAG retrieval      | ~0.05 s |
| Llama3 generation  | ~1-2 s  |
| Text-to-speech     | ~0.3 s  |

Total response time:

~2 seconds.

---

# Future Improvements

Possible extensions include:

* multi-document knowledge bases
* GPU acceleration for speech models
* improved conversation memory

---

# License

This project is intended for educational and research purposes.
