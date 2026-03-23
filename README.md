#  Llama3 RAG Voice Assistant

A **local AI voice assistant** that combines **RAG (Retrieval-Augmented Generation)** with **speech-to-text and text-to-speech**, allowing users to ask questions using voice and receive intelligent spoken answers.

---

##  Features

*  **Voice Input (Speech-to-Text)**

  * Powered by `faster-whisper`
  * Fast and fully local

*  **RAG (Retrieval-Augmented Generation)**

  * Supports **PDF, Excel, and CSV files**
  * Hybrid retrieval:

    * FAISS (semantic search)
    * BM25 (keyword search)
  * Reranking with CrossEncoder for better accuracy

*  **LLM (Llama3 via Ollama)**

  * Answers questions using:

    * Your uploaded documents
    * General knowledge when needed

*  **Text-to-Speech (TTS)**

  * Powered by `Piper`
  * Fully local and fast
  * Auto-plays responses

*  **Conversation Memory**

  * Keeps chat history during session

---

##  How It Works

```
User Voice
   ↓
Whisper (Speech → Text)
   ↓
Hybrid RAG Retrieval (FAISS + BM25)
   ↓
Reranker (BGE CrossEncoder)
   ↓
Llama3 (Ollama)
   ↓
Answer
   ↓
Piper (Text → Speech)
```

---

##  Supported File Types

Place your documents inside:

```
data/documents/
```

Supported formats:

* `.pdf`
* `.xlsx`
* `.csv`

---

##  Smart Data Handling

* Large datasets are optimized by:

  * Selecting important columns
  * Limiting rows for performance
* Automatically adapts to different file structures

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/fouad-2004/STT-TTS-voice-Assistant.git
cd STT-TTS-voice-Assistant
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Install Ollama & Llama3

### Install Ollama:

https://ollama.com/

### Pull Llama3:

```bash
ollama pull llama3
```

---

##  Install Piper (TTS)

Download a voice model and place it in:

```
voices/voice.onnx
voices/voice.onnx.json
```

---

##  Run the App

```bash
streamlit run app.py
```

---

##  Example Questions

* "What is machine learning?"
* "Which ship is closest to the port of Los Angeles?"
* "What cargo is vessel X carrying?"
* "Summarize the uploaded PDF"

---

##  Configuration

### LLM settings (in `rag_utils.py`):

```python
temperature = 0.0  # deterministic answers
```

---

##  Performance Tips

* Keep datasets under **50MB** for best performance
* Use column filtering for large CSV/Excel files
* First run may take longer (vector DB creation)

---

##  Tech Stack

* **Frontend:** Streamlit
* **LLM:** Llama3 (Ollama)
* **Embeddings:** BGE (HuggingFace)
* **Vector DB:** FAISS
* **Keyword Search:** BM25
* **Reranker:** CrossEncoder (BGE)
* **STT:** faster-whisper
* **TTS:** Piper

---

##  Known Limitations

* Very large datasets may slow down indexing
* Analytical queries (e.g., "closest", "largest") rely on LLM reasoning
* GPU support may vary depending on hardware

---

##  Future Improvements

*  Smart tool usage (auto calculations)
*  Real-time streaming voice responses

##  Project Highlights

* Fully **local AI assistant**
* Combines **RAG + Voice + LLM**
* Production-style architecture
* Real-world AI system design

##  License

This project is for educational purposes.
