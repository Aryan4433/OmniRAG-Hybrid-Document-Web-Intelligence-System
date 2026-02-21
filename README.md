# 🧠 OmniRAG – Hybrid Document & Web Intelligence System

OmniRAG is an advanced **Retrieval-Augmented Generation (RAG)** application that combines document-based semantic search with real-time web retrieval.  

The system first attempts to answer queries using context extracted from uploaded PDF documents. If the answer is not found in the document, it automatically falls back to live web search using DuckDuckGo and generates a response using LLM reasoning.

This hybrid architecture ensures higher reliability, reduced hallucination, and dynamic knowledge augmentation.

---

## 🚀 Features

### 📄 Document-Based Semantic Search
- Upload and process PDF documents
- Recursive text chunking with overlap
- SentenceTransformer embeddings
- Chroma vector database for similarity search

### 🔎 Intelligent Web Fallback
- Detects when answer is not present in document
- Performs real-time DuckDuckGo search
- Synthesizes web results using LLM reasoning

### 🤖 LLM-Powered Generation
- Groq LLaMA 3.1 8B Instant model
- Context-constrained RAG prompting
- Explicit hallucination control using `NOT_FOUND` logic

### 💬 Interactive Chat Interface
- Built with Streamlit
- Persistent session state
- Multi-turn conversational experience

---

## 🏗️ System Architecture


🛠️ Tech Stack

LangChain

Groq (LLaMA 3.1 8B Instant)

Chroma Vector Store

SentenceTransformers

DuckDuckGo Search Tool

Streamlit

Python

Persistent session state

Multi-turn interaction support
