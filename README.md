# OmniRAG-Hybrid-Document-Web-Intelligence-System
OmniRAG is an advanced Retrieval-Augmented Generation (RAG) application that intelligently combines document-based semantic search with real-time web retrieval.

The system first attempts to answer queries using context extracted from uploaded PDF documents. If the requested information is not found in the document context, it automatically falls back to live web search using DuckDuckGo and generates a response using LLM reasoning.

This architecture ensures:

High accuracy from local knowledge sources

Reduced hallucination

Real-time knowledge augmentation when required

Intelligent fallback logic

🚀 Key Features
📄 Document-Based Semantic Search

Upload PDF documents

Automatic text chunking with overlap

Vector embedding generation using SentenceTransformers

Chroma vector database for similarity retrieval

🔎 Intelligent Web Fallback

Automatic detection when answer is not found in document

Real-time DuckDuckGo search integration

LLM-powered synthesis of web results

🤖 LLM-Powered Reasoning

Groq LLaMA 3.1 8B model integration

Context-constrained RAG prompting

Explicit hallucination control using NOT_FOUND logic

💬 Interactive Chat Interface

Streamlit-based conversational UI


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
