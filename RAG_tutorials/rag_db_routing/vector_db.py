import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st
from config import  COLLECTIONS
from langchain_community.vectorstores import Qdrant

def initialize_models():
    """Initialize OpenAI models and Qdrant client"""
    if (st.session_state.openai_api_key and
            st.session_state.qdrant_url):

        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.llm = ChatOpenAI(temperature=0)

        try:
            client = QdrantClient(
                url=st.session_state.qdrant_url,
                # api_key=st.session_state.qdrant_api_key
            )

            # Test connection
            client.get_collections()
            vector_size = 1536
            st.session_state.databases = {}
            for db_type, config in COLLECTIONS.items():
                try:
                    client.get_collection(config.collection_name)
                except Exception:
                    # Create collection if it doesn't exist
                    client.create_collection(
                        collection_name=config.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )

                st.session_state.databases[db_type] = Qdrant(
                    client=client,
                    collection_name=config.collection_name,
                    embeddings=st.session_state.embeddings
                )

            return True
        except Exception as e:
            st.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    return False
