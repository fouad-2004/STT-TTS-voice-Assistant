import os
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama


# =====================================
# Load Embeddings
# =====================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )


# =====================================
# Load Documents (PDF + Excel)
# =====================================

def load_documents():

    docs = []
    folder = "data/documents"

    if not os.path.exists(folder):
        return docs

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        # PDF files
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # Excel files
        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)

            # convert rows to documents
            loader = DataFrameLoader(df, page_content_column=df.columns[0])
            docs.extend(loader.load())

    return docs


# =====================================
# Split Documents into Chunks
# =====================================

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# =====================================
# Load / Create Vector Store
# =====================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()
    vector_path = "data/vector_store"

    # Load existing vector DB
    if os.path.exists(vector_path):
        return FAISS.load_local(
            vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create new vector DB
    docs = load_documents()
    chunks = split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(vector_path)

    return vector_store


# =====================================
# Load LLM (Llama3 via Ollama)
# =====================================

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3",
        temperature=0.2
    )


# =====================================
# Answer Question (RAG)
# =====================================

def answer_question(query):

    vector_store = load_vector_store()
    llm = load_llm()

    # Retrieve relevant docs
    docs = vector_store.similarity_search(query, k=5)

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
You are an AI assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content