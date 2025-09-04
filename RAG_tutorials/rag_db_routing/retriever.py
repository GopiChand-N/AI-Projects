
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
from langchain_community.vectorstores import Qdrant



def query_database(db: Qdrant, question: str) -> tuple[str, list]:
    """Query the database and return answer and relevant documents"""
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        relevant_docs = retriever.get_relevant_documents(question)

        if relevant_docs:
            # Use simpler chain creation with hub prompt
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on provided context.
                             Always be direct and concise in your responses.
                             If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
                             Base your answers strictly on the provided context and avoid making assumptions."""),
                ("human", "Here is the context:\n{context}"),
                ("human", "Question: {input}"),
                ("assistant", "I'll help answer your question based on the context provided."),
                ("human", "Please provide your answer:"),
            ])
            combine_docs_chain = create_stuff_documents_chain(st.session_state.llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = retrieval_chain.invoke({"input": question})
            return response['answer'], relevant_docs

        raise ValueError("No relevant documents found in database")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try rephrasing your question.", []

