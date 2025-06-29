import streamlit as st
import importlib
import subprocess
import sys
import os
import time

# Show spinner while checking/installing packages
with st.spinner('Checking and installing required packages, please wait...'):
    required_packages = [
        "streamlit",
        "PyPDF2",
        "langchain",
        "langchain-community",
        "requests",
        "pydantic",
        "sentence-transformers",
        "faiss-cpu",
        "python-docx",
    ]

    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from typing import Optional, List
from pydantic import BaseModel, Field
import requests

class GroqLLM(LLM, BaseModel):
    api_key: str
    model: str = Field(default="llama3-70b-8192")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        for _ in range(3):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                time.sleep(10)
            else:
                raise ValueError(f"Groq API error {response.status_code}: {response.text}")
        raise ValueError("Rate limit exceeded. Please try again later.")

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = [Generation(text=self._call(prompt, stop=stop)) for prompt in prompts]
        return LLMResult(generations=[generations])


st.markdown(
    """
    <h2 style='font-size:28px; white-space: nowrap;'>
        📄 Document Chatbot with Groq LLaMA3
    </h2>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

if not uploaded_file:
    st.info("Please upload a PDF or DOCX document to begin.")
    st.stop()

file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
full_text = ""

try:
    if file_ext == ".pdf":
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text
    elif file_ext == ".docx":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            full_text += para.text + "\n"
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        st.stop()

    if not full_text.strip():
        st.error("No text could be extracted from the document.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    chunks = [c for c in chunks if len(c.strip()) > 100]

    if not chunks:
        st.error("Extracted content is too short to analyze.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

except Exception as e:
    st.error(f"Error reading the document: {e}")
    st.stop()

question = st.text_input("Ask a question about the document:")

if question:
    try:
        docs = vector_store.similarity_search(question, k=8)
        if not docs:
            st.warning("No relevant sections found in the document.")
        else:
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
            )
            llm = GroqLLM(api_key=st.secrets["api_keys"]["groq"])
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

            with st.spinner("Generating answer..."):
                answer = chain.run({"input_documents": docs, "question": question})

            st.subheader("Answer:")
            st.write(answer)
    except Exception as e:
        st.error(f"Error generating answer: {e}")
