import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun



load_dotenv()


# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# streamlit Page setup 
st.set_page_config(page_title = "Hybrid-RAG-Assistant")
st.title("RAG Chatbot with Dynamic Web Retrieval")

# sessionstate

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history

for msg in st.session_state.messages:
    st.chat_message(msg['role']).markdown(msg['content'])


# Upload file

Uploaded_file = st.file_uploader("Uploaded_pdf",type="pdf")

#create vector_store function

def create_vector_store(file):
    with open("temp.pdf","wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    Splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap= 200
    )
    split_docs = Splitter.split_documents(documents)

    embeddings= SentenceTransformerEmbeddings(
        model_name = "all-MiniLM-L12-v2"
    )

    return Chroma.from_documents(split_docs,embeddings)


if Uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document: "):
        st.session_state.vectorestore = create_vector_store(Uploaded_file)
    st.success("Document Passed Sucessfully!")


# --User Input--

prompt = st.chat_input("Ask something............")


if prompt:

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':"user", "content":prompt})

    llm= ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY") 
    )

    answer = None


    # -- use RAG first -- 
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in docs])

        if context.strip():
            rag_prompt = f"""
Answer ONLY from the document context below.
If the answer is not present, respond exactly with: NOT_FOUND

Context:
{context}

Question:
{prompt}
"""
            rag_response = llm.invoke(rag_prompt).content

            if "NOT_FOUND" not in rag_response:
                answer = rag_response

    # ---------------- DUCKDUCKGO FALLBACK ----------------
    if not answer:
        search = DuckDuckGoSearchRun()
        web_results = search.invoke(prompt)

        web_prompt = f"""
Answer using the web search results below.

Web Results:
{web_results}

Question:
{prompt}
"""
        answer = llm.invoke(web_prompt).content

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


























