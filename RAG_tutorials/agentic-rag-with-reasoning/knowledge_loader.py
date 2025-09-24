from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase

def load_knowledge(openai_key, recreate=True, upsert=False, skip_existing=True):
    """
    Load and initialize the knowledge base with a vector database.

    Parameters:
        openai_key (str): OpenAI API key used to generate embeddings.
        recreate (bool): Whether to recreate the vector database from scratch.
        upsert (bool): Whether to update existing documents in the database.
        skip_existing (bool): Whether to skip already loaded documents.

    Returns:
        UrlKnowledge: An instance of the knowledge base.
    """
    kb = PDFUrlKnowledgeBase(
        urls=["https://docs.agno.com/introduction/agents.md"],  # Default knowledge source
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="agno_docs",
            search_type=SearchType.vector,
            embedder=OpenAIEmbedder(api_key=openai_key)
        )
    )
    kb.load(recreate=recreate, upsert=upsert, skip_existing=skip_existing)
    return kb