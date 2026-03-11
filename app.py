import streamlit as st
import os
import base64

from audio_utils import text_to_speech, speech_to_text_whisper
from rag_utils import answer_question, load_vector_store


# ======================================
# Page Configuration
# ======================================
st.set_page_config(page_title="Llama3 RAG Voice Assistant", layout="wide")
st.title(" Llama3 RAG Voice Assistant")


# ======================================
# Paths
# ======================================
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ======================================
# Session State
# ======================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()


# ======================================
# Question Processor
# ======================================
def process_question(question: str):

    answer = answer_question(
        vector_store=st.session_state.vector_store,
        question=question,
        chat_history=st.session_state.chat_history
    )

    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    return answer

# ======================================
# MICROPHONE INPUT
# ======================================
st.divider()
st.subheader("Ask a Question (Voice)")

with st.form("mic_form"):
    audio_bytes = st.audio_input("Record your question")
    submit_mic = st.form_submit_button("Ask")

if submit_mic and audio_bytes:

    audio_path = os.path.join(UPLOAD_DIR, "mic_input.wav")

    with open(audio_path, "wb") as f:
        f.write(audio_bytes.getbuffer())

    with st.spinner("Transcribing audio..."):
        spoken_text = speech_to_text_whisper(audio_path)

    st.success("Speech recognized")
    st.write("You said:", spoken_text)

    with st.spinner("Llama3 thinking..."):
        answer = process_question(spoken_text)

    st.markdown("### Answer")
    st.write(answer)

    audio_file = text_to_speech(answer)

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)


# ======================================
# Conversation History
# ======================================
st.divider()
st.subheader("Conversation")

for msg in st.session_state.chat_history:
    role = msg["role"].capitalize()
    content = msg["content"]

    st.markdown(f"**{role}:** {content}")