from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain_core.prompts import PromptTemplate
import pprint
import nest_asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tempfile, os

nest_asyncio.apply()

retriever = None


def initialize_session_state():
    """Initialize session state variables for API keys and URLs."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        # Initialize API keys and URLs
        st.session_state.openai_api_key = ""
        st.session_state.qdrant_api_key = ""
        st.session_state.qdrant_url = "http://localhost:6333"

def setup_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")

        st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key,
                                                        type="password")

        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url)


        if not all([st.session_state.openai_api_key, st.session_state.qdrant_url]):
            st.warning("Please provide the required API keys and URLs")
            st.stop()

        st.session_state.initialized = True


initialize_session_state()
import hashlib
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "index_key" not in st.session_state:
    st.session_state.index_key = None
setup_sidebar()

# Use session state variables instead of config
openai_api_key = st.session_state.openai_api_key

# Update embeddings initialization
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=st.session_state.openai_api_key
)

# Update Qdrant client initialization
client = QdrantClient(
    url=st.session_state.qdrant_url,
    # api_key=st.session_state.qdrant_api_key
)




def web_search(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    ret = st.session_state.get("retriever")
    if ret is None:
        return {"keys": {"documents": documents, "question": question}}
    new_docs = ret.get_relevant_documents(question)
    documents.extend(new_docs)
    return {"keys": {"documents": documents, "question": question}}

import bs4
def load_documents(file_or_url: str, is_url: bool = True) -> list:
    try:
        if is_url:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_or_url)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []


st.subheader("Document Input")

input_option = st.radio(
    "Choose input method:",
    ["URL", "File Upload"],
    key="input_option",
)

with st.form("docs"):
    url = ""
    uploaded_file = None

    if input_option == "URL":
        url = st.text_input("Enter document URL:", value="", placeholder="only .pdf, .txt, .md files")
    else:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            key="file_input",
        )

    submitted = st.form_submit_button("Load & index")

docs = None
if submitted:
    # Load docs only when the button is clicked (prevents reloading while typing a question)
    if st.session_state.input_option == "URL" and url:
        docs = load_documents(url, is_url=True)
    elif uploaded_file is not None:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
            st.success(f"Loaded {len(docs)} documents from the uploaded file.")
        # Clean up the temporary file
        os.unlink(tmp_file.name)

if docs:
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)
    st.success(f"Document split into {len(all_splits)} chunks.")

    # Build a stable content hash so we only (re)index when the document really changed
    doc_key = hashlib.md5("".join(d.page_content for d in all_splits).encode()).hexdigest()

    if st.session_state.index_key != doc_key:
        # Fresh (re)index
        collection_name = "rag-qdrant"

        try:
            QdrantClient(url=st.session_state.qdrant_url).delete_collection(collection_name)
        except Exception:
            pass

        client = QdrantClient(url=st.session_state.qdrant_url)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        vectorstore.add_documents(all_splits)
        st.success("Documents added to the vectorstore.")

        # Save retriever & doc key in session so it survives reruns
        st.session_state.retriever = vectorstore.as_retriever()
        st.session_state.index_key = doc_key
        st.success("Index ready. You can type/edit your question without re-indexing.")
    else:
        st.info("No changes detected in the document. Reusing the existing index.")



class GraphState(TypedDict):
    keys: Dict[str, any]



def retrieve(state):
    print("~-retrieve-~")
    state_dict = state["keys"]
    question = state_dict["question"]

    ret = st.session_state.get("retriever")
    if ret is None:
        return {"keys": {"documents": [], "question": question}}

    documents = ret.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    prompt = PromptTemplate(template="""Based on the following context, please answer the question.
        Context: {context}
        Question: {question}
        Answer:""", input_variables=["context", "question"])
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=st.session_state.openai_api_key, temperature=0, max_tokens=1000)
    context = "\n\n".join(doc.page_content for doc in documents)
    rag_chain = ({"context": lambda x: context, "question": lambda x: question} | prompt | llm | StrOutputParser())
    generation = rag_chain.invoke({})
    return {"keys": {"documents": documents, "question": question, "generation": generation}}


def grade_documents(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=st.session_state.openai_api_key, temperature=0, max_tokens=200)
    prompt = PromptTemplate(template="""Return ONLY a JSON object with {"score":"yes"} or {"score":"no"}.
        Document: {context}
        Question: {question}""", input_variables=["context","question"])
    chain = prompt | llm | StrOutputParser()
    filtered_docs = []
    search = "No"
    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            import re, json
            json_match = re.search(r'\{.*\}', response)
            score = json.loads(json_match.group() if json_match else response)
            if score.get("score") == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
        except Exception:
            filtered_docs.append(d)
    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    prompt = PromptTemplate(template="""Generate a search-optimized version of this question... Return only the improved question:""", input_variables=["question"])
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=st.session_state.openai_api_key, temperature=0, max_tokens=200)
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    return {"keys": {"documents": state_dict["documents"], "question": better_question}}

def decide_to_generate(state):
    print("~-decide to generate-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":

        print("~-decision: transform query and run web search-~")
        return "transform_query"
    else:
        print("~-decision: generate-~")
        return "generate"


def format_document(doc: Document) -> str:
    return f"""
    Source: {doc.metadata.get('source', 'Unknown')}
    Title: {doc.metadata.get('title', 'No title')}
    Content: {doc.page_content[:200]}...
    """


def format_state(state: dict) -> str:
    formatted = {}

    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value

    return formatted


workflow = StateGraph(GraphState)

# Define the nodes by langgraph
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

st.title("🔄 Corrective RAG Agent")

st.text("A possible query: What are the experiment results and ablation studies in this research paper?")

# User input
user_question = st.text_input("Please enter your question:", key="question")

if st.button("Ask"):
    inputs = {"keys": {"question": st.session_state.question}}

    for output in app.stream(inputs):
        for key, value in output.items():
            print("key", key)
            print("value", value)
            with st.expander(f"Step '{key}':"):
                st.text(pprint.pformat(format_state(value["keys"]), indent=2, width=80))

    final_generation = value['keys'].get('generation', 'No final generation produced.')
    st.subheader("Final Generation:")
    st.write(final_generation)