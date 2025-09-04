import streamlit as st
from session_state import init_session_state
from vector_db import initialize_models
from config import  COLLECTIONS
from doc_loader import process_document
from router import route_query
from agent import _handle_web_fallback
from retriever import query_database

init_session_state()

PERSIST_DIRECTORY = "db_storage"

def main():
    """Main application function."""
    st.set_page_config(page_title="RAG Agent with Database Routing", page_icon="📚")
    st.title("📠 RAG Agent with Database Routing")

    # Sidebar for API keys and configuration
    with st.sidebar:
        st.header("Configuration")

        # OpenAI API Key
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            key="api_key_input"
        )

        # Qdrant Configuration
        qdrant_url = st.text_input(
            "Enter Qdrant URL:",
            value=st.session_state.qdrant_url,
            help="Example: https://your-cluster.qdrant.tech"
        )

        # qdrant_api_key = st.text_input(
        #     "Enter Qdrant API Key:",
        #     type="password",
        #     value=st.session_state.qdrant_api_key
        # )

        # Update session state
        if api_key:
            st.session_state.openai_api_key = api_key
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        # if qdrant_api_key:
        #     st.session_state.qdrant_api_key = qdrant_api_key

        # Initialize models if all credentials are provided
        if (st.session_state.openai_api_key and
                st.session_state.qdrant_url):
            if initialize_models():
                st.success("Connected to OpenAI and Qdrant successfully!")
            else:
                st.error("Failed to initialize. Please check your credentials.")
        else:
            st.warning("Please enter all required credentials to continue")
            st.stop()

        st.markdown("---")

    st.header("Document Upload")
    st.info("Upload documents to populate the databases. Each tab corresponds to a different database.")
    tabs = st.tabs([collection_config.name for collection_config in COLLECTIONS.values()])

    for (collection_type, collection_config), tab in zip(COLLECTIONS.items(), tabs):
        with tab:
            st.write(collection_config.description)
            uploaded_files = st.file_uploader(
                f"Upload PDF documents to {collection_config.name}",
                type="pdf",
                key=f"upload_{collection_type}",
                accept_multiple_files=True
            )

            if uploaded_files:
                with st.spinner('Processing documents...'):
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        texts = process_document(uploaded_file)
                        all_texts.extend(texts)

                    if all_texts:
                        db = st.session_state.databases[collection_type]
                        db.add_documents(all_texts)
                        st.success("Documents processed and added to the database!")

    # Query section
    st.header("Ask Questions")
    st.info("Enter your question below to find answers from the relevant database.")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner('Finding answer...'):
            # Route the question
            collection_type = route_query(question)

            if collection_type is None:
                # Use web search fallback directly
                answer, relevant_docs = _handle_web_fallback(question)
                st.write("### Answer (from web search)")
                st.write(answer)
            else:
                # Display routing information and query the database
                st.info(f"Routing question to: {COLLECTIONS[collection_type].name}")
                db = st.session_state.databases[collection_type]
                answer, relevant_docs = query_database(db, question)
                st.write("### Answer")
                st.write(answer)


if __name__ == "__main__":
    main()