import streamlit as st



def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'databases' not in st.session_state:
        st.session_state.databases = {}