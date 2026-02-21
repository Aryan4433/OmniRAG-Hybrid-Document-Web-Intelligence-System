# ============================================
#              IMPORTS
# ============================================

import os
import streamlit as st
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# RAG Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Agent Components
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

# Web Tool
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================
#           LOAD ENV VARIABLES
# ============================================

load_dotenv()

# ============================================
#           STREAMLIT CONFIG
# ============================================

st.set_page_config(page_title="OmniRAG Agent", layout="wide")
st.title("🧠 OmniRAG – Agentic RAG System")

# ============================================
#           SESSION STATE
# ============================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
#        DISPLAY CHAT HISTORY
# ============================================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ============================================
#           PDF UPLOAD
# ============================================

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

def create_vectorstore(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = create_vectorstore(uploaded_file)
    st.success("✅ Document processed successfully!")


# ============================================
#             USER INPUT
# ============================================

user_prompt = st.chat_input("Ask a question...")

if user_prompt:

    # Show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # ============================================
    #             CREATE LLM
    # ============================================

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # ============================================
    #             CREATE TOOLS
    # ============================================

    tools = []

    # ---- Web Tool (Always Available) ----
    search = DuckDuckGoSearchRun()

    web_tool = Tool(
        name="Web_Search",
        func=search.run,
        description="Use this tool for answering questions about current events or information not found in the uploaded PDF."
    )

    tools.append(web_tool)

    # ---- PDF Tool (Only if PDF Uploaded) ----
    if st.session_state.vectorstore:

        def pdf_search(query: str):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            return "\n\n".join([d.page_content for d in docs])

        pdf_tool = Tool(
            name="PDF_Search",
            func=pdf_search,
            description="Use this tool to search information from the uploaded PDF document."
        )

        tools.append(pdf_tool)

    # ============================================
    #         CREATE REACT AGENT
    # ============================================

    prompt_template = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # ============================================
    #           RUN AGENT
    # ============================================

    with st.spinner("Thinking..."):

        response = agent_executor.invoke({"input": user_prompt})
        answer = response["output"]

    # Show assistant response
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})