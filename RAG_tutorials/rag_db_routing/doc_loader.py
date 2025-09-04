from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os
from typing import List, Dict, Any, Literal, Optional
from langchain_core.documents import Document
import tempfile

def process_document(file) -> List[Document]:
    """Process uploaded PDF document"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Clean up temporary file
        os.unlink(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        return texts
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []
