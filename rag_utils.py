import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from rank_bm25 import BM25Okapi


# ======================================
# Load Embeddings
# ======================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )


# ======================================
# Load Vector Store
# ======================================
@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()

    vector_path = "data/vector_store"

    if os.path.exists(vector_path):

        vector_store = FAISS.load_local(
            vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store

    return None


# ======================================
# Load LLM (Llama3)
# ======================================
def load_llm():

    llm = ChatOllama(
        model="llama3",
        temperature=0.2,
        num_predict=300
    )

    return llm


# ======================================
# Load Reranker
# ======================================
@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

reranker = load_reranker()


# ======================================
# Build Chat History
# ======================================
def build_history(chat_history):

    if not chat_history:
        return ""

    history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in chat_history[-4:]
    )

    return history


# ======================================
# Hybrid Retrieval (Vector + BM25)
# ======================================
def hybrid_retrieval(vector_store, question):

    # Vector search
    vector_docs = vector_store.max_marginal_relevance_search(
        question,
        k=6,
        fetch_k=12
    )

    # Prepare BM25
    corpus = [doc.page_content for doc in vector_docs]

    tokenized_corpus = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = question.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(bm25_scores, vector_docs),
        key=lambda x: x[0],
        reverse=True
    )

    docs = [doc for score, doc in ranked]

    return docs


# ======================================
# Rerank Documents
# ======================================
def rerank_documents(question, docs):

    pairs = [(question, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    top_docs = [doc for score, doc in ranked[:4]]

    return top_docs


# ======================================
# Build Context
# ======================================
def build_context(docs):

    context = "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    return context


# ======================================
# Main RAG Function
# ======================================
def answer_question(vector_store, question, chat_history=None):

    llm = load_llm()

    history_text = build_history(chat_history)

    if vector_store is None:

        prompt = f"""
You are a helpful AI assistant.

Conversation history:
{history_text}

User question:
{question}

Answer clearly and concisely.
"""

        response = llm.invoke(prompt)

        return response.content if hasattr(response, "content") else str(response)


    # Hybrid retrieval
    docs = hybrid_retrieval(vector_store, question)


    # Neural reranking
    docs = rerank_documents(question, docs)


    # Build context
    context = build_context(docs)


    # RAG prompt
    prompt = f"""
You are a knowledgeable assistant.

Use ONLY the provided context to answer the question.

If the answer is not contained in the context,
say the information was not found in the documents.

Conversation history:
{history_text}

Context:
{context}

Question:
{question}

Provide a clear and concise answer.
"""


    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)