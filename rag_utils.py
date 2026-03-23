import os
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from langchain_ollama import ChatOllama

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


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
# Load Documents (PDF + Excel + CSV)
# =====================================

def load_documents():

    docs = []
    folder = "data/documents"

    if not os.path.exists(folder):
        return docs

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        # PDF
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # Excel
        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)

            target_cols = ["BaseDateTime", "LAT", "LON", "VesselName", "IMO", "Cargo"]

            # Case 1: Large structured dataset
            if all(col in df.columns for col in target_cols):

                df = df[target_cols]
                df = df.head(2000)

                for _, row in df.iterrows():
                    text = (
                        f"DateTime: {row['BaseDateTime']}, "
                        f"Latitude: {row['LAT']}, "
                        f"Longitude: {row['LON']}, "
                        f"Vessel: {row['VesselName']}, "
                        f"IMO: {row['IMO']}, "
                        f"Cargo: {row['Cargo']}"
                    )
                    docs.append(Document(page_content=text))

            # Case 2: Any other Excel file
            else:
                df = df.head(1000)

                for _, row in df.iterrows():
                    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                    docs.append(Document(page_content=text))


        # CSV
        elif file.endswith(".csv"):
            df = pd.read_csv(path)

            target_cols = ["BaseDateTime", "LAT", "LON", "VesselName", "IMO", "Cargo"]

            # Case 1: Large structured dataset
            if all(col in df.columns for col in target_cols):

                df = df[target_cols]
                df = df.head(2000)

                for _, row in df.iterrows():
                    text = (
                        f"DateTime: {row['BaseDateTime']}, "
                        f"Latitude: {row['LAT']}, "
                        f"Longitude: {row['LON']}, "
                        f"Vessel: {row['VesselName']}, "
                        f"IMO: {row['IMO']}, "
                        f"Cargo: {row['Cargo']}"
                    )
                    docs.append(Document(page_content=text))

            # Case 2: Any other CSV file
            else:
                df = df.head(1000)

                for _, row in df.iterrows():
                    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                    docs.append(Document(page_content=text))

    return docs


# =====================================
# Split Documents
# =====================================

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# =====================================
# Load Chunks (cached)
# =====================================

@st.cache_resource
def load_chunks():

    docs = load_documents()
    chunks = split_documents(docs)

    return chunks


# =====================================
# Build BM25
# =====================================

@st.cache_resource
def build_bm25(chunks):

    texts = [doc.page_content for doc in chunks]
    tokenized = [text.split() for text in texts]

    bm25 = BM25Okapi(tokenized)

    return bm25, texts


# =====================================
# Load Vector Store
# =====================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()
    vector_path = "data/vector_store"
    index_file = os.path.join(vector_path, "index.faiss")

    # ✅ Only load if FAISS index actually exists
    if os.path.exists(index_file):
        return FAISS.load_local(
            vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ✅ Otherwise create it
    chunks = load_chunks()

    if len(chunks) == 0:
        raise ValueError("No documents found in data/documents")

    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(vector_path, exist_ok=True)
    vector_store.save_local(vector_path)

    return vector_store

# =====================================
# Load Reranker
# =====================================

@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base",
                        device="cpu"
                        )


# =====================================
# Hybrid Retrieval + Reranking
# =====================================

def hybrid_search(query, k=5):

    vector_store = load_vector_store()
    chunks = load_chunks()

    # FAISS search
    faiss_docs = vector_store.similarity_search(query, k=10)

    # BM25 search
    bm25, texts = build_bm25(chunks)
    scores = bm25.get_scores(query.split())

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    bm25_docs = [chunks[i] for i in top_indices]

    # Combine
    combined = faiss_docs + bm25_docs

    # Remove duplicates
    unique_docs = []
    seen = set()

    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # =========================
    # RERANK
    # =========================

    reranker = load_reranker()

    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, unique_docs),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    return ranked_docs[:k]


# =====================================
# Load LLM
# =====================================

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3",
        temperature=0.2
    )


# =====================================
# Answer Question
# =====================================

def answer_question(query):

    llm = load_llm()

    docs = hybrid_search(query, k=15)
    docs = sorted(docs, key=lambda x: x.page_content)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an intelligent AI assistant.

    You have access to the following context extracted from documents:

    ---------------------
    {context}
    ---------------------

    Instructions:
    - Answer the question using the provided data when relevant.
    - If the question requires comparison, calculation, or reasoning (e.g., closest, largest, highest), analyze ALL the data before answering.
    - Do NOT rely on a single entry — compare multiple entries if needed.
    - Return ONLY the final result, not the full list.
    - Keep the answer short and direct (1–2 sentences).

    - If the data is not sufficient, answer using your general knowledge.


    Question:
    {query}

    Answer:
    """
    response = llm.invoke(prompt)

    return response.content