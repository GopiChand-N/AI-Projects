import os
import asyncio
from typing import List, Optional, AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, AnyHttpUrl
from dotenv import load_dotenv

# --- Agno / RAG pieces ---
from agno.agent import Agent, RunEvent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType

# ---------- App bootstrap ----------
load_dotenv()

DEFAULT_URLS = [
    "https://docs.agno.com/introduction/agents.md"
]
LANCEDB_URI = os.getenv("LANCEDB_URI", "tmp/lancedb")
LANCEDB_TABLE = os.getenv("LANCEDB_TABLE", "agno_docs")

app = FastAPI(title="Agentic RAG with Reasoning (FastAPI)", version="1.0.0")

# CORS (adjust for your frontend origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- App state & helpers ----------

class AppState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.knowledge: Optional[UrlKnowledge] = None
        self.agent: Optional[Agent] = None
        self.urls: List[str] = list(DEFAULT_URLS)
        self.api_key: Optional[str] = None

    def reset(self):
        self.knowledge = None
        self.agent = None

STATE = AppState()

def get_effective_api_key(header_key: Optional[str]) -> str:
    """
    API key precedence:
    1) X-OpenAI-Key header on each request (optional override)
    2) Previously configured key via /configure
    3) OPENAI_API_KEY env var
    """
    key = header_key or STATE.api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="Missing OpenAI API key. Provide it via X-OpenAI-Key header, /configure, or environment variable OPENAI_API_KEY.")
    return key

async def ensure_initialized(api_key: str) -> None:
    """
    Initializes knowledge base and agent if not already ready.
    Safe for concurrent calls.
    """
    if STATE.agent and STATE.knowledge and STATE.api_key == api_key:
        return

    async with STATE.lock:
        # double-check inside lock
        if STATE.agent and STATE.knowledge and STATE.api_key == api_key:
            return

        os.makedirs(LANCEDB_URI, exist_ok=True)

        # Initialize knowledge
        kb = UrlKnowledge(
            urls=STATE.urls,
            vector_db=LanceDb(
                uri=LANCEDB_URI,
                table_name=LANCEDB_TABLE,
                search_type=SearchType.vector,
                embedder=OpenAIEmbedder(api_key=api_key),
            ),
        )
        # Recreate when key changed or first boot to avoid index mismatch
        kb.load(recreate=True)

        # Initialize agent
        agent = Agent(
            model=OpenAIChat(
                id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=api_key
            ),
            knowledge=kb,
            search_knowledge=True,
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Include sources (URLs) in your response.",
                "Always search your knowledge before answering the question.",
            ],
            markdown=True,
        )

        STATE.knowledge = kb
        STATE.agent = agent
        STATE.api_key = api_key

# ---------- Schemas ----------

class ConfigureRequest(BaseModel):
    openai_api_key: Optional[str] = None
    urls: Optional[List[AnyHttpUrl]] = None
    recreate: bool = False  # force a fresh index load

class AddUrlRequest(BaseModel):
    url: AnyHttpUrl

class AskRequest(BaseModel):
    query: str
    show_full_reasoning: bool = True

class Citation(BaseModel):
    title: Optional[str] = None
    url: AnyHttpUrl

class AskResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    citations: List[Citation] = []

# ---------- Routes ----------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/configure")
async def configure(req: ConfigureRequest):
    """
    Configure default API key and seed URLs.
    This recreates the vector DB if 'recreate' is True or if the key changes.
    """
    if req.urls is not None:
        # keep as str list for UrlKnowledge
        STATE.urls = [str(u) for u in req.urls]

    if req.openai_api_key:
        # If the key changed, reset to force rebuild
        if STATE.api_key != req.openai_api_key:
            STATE.api_key = req.openai_api_key
            STATE.reset()

    # If we already had key (via env or set above), optionally rebuild now
    api_key = STATE.api_key or os.getenv("OPENAI_API_KEY")
    if api_key:
        if req.recreate:
            STATE.reset()
        await ensure_initialized(api_key)

    return {
        "configured_urls": STATE.urls,
        "has_api_key": bool(STATE.api_key or os.getenv("OPENAI_API_KEY")),
        "recreated": req.recreate
    }

@app.get("/sources")
async def list_sources():
    return {"urls": STATE.urls}

@app.post("/add_url")
async def add_url(req: AddUrlRequest, x_openai_key: Optional[str] = Header(default=None)):
    api_key = get_effective_api_key(x_openai_key)
    await ensure_initialized(api_key)

    # Upsert into KB
    if not STATE.knowledge:
        raise HTTPException(status_code=500, detail="Knowledge base not initialized.")
    if str(req.url) not in STATE.urls:
        STATE.urls.append(str(req.url))
    try:
        # incremental load
        STATE.knowledge.urls = STATE.urls
        STATE.knowledge.load(recreate=False, upsert=True, skip_existing=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load URL: {e}")

    return {"added": str(req.url), "total_urls": len(STATE.urls)}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_openai_key: Optional[str] = Header(default=None)):
    """
    Synchronous Q&A that internally streams events, aggregates them,
    and returns a single JSON result with answer, reasoning (if available), and citations.
    """
    api_key = get_effective_api_key(x_openai_key)
    await ensure_initialized(api_key)

    if not STATE.agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    citations: List[Dict[str, Any]] = []
    answer_text: str = ""
    reasoning_text: Optional[str] = None

    try:
        for chunk in STATE.agent.run(
            req.query,
            stream=True,
            show_full_reasoning=req.show_full_reasoning,
            stream_intermediate_steps=True,
        ):
            # Reasoning (if provided by the model/tooling)
            rc = getattr(chunk, "reasoning_content", None)
            if rc:
                reasoning_text = rc  # keep latest

            # Accumulate answer when final content comes through
            content = getattr(chunk, "content", None)
            if content and getattr(chunk, "event", None) in {RunEvent.run_response_content, RunEvent.run_completed}:
                if isinstance(content, str):
                    answer_text += content

            # Collect citations (if present)
            citations_attr = getattr(chunk, "citations", None)
            if citations_attr and getattr(citations_attr, "urls", None):
                # chunk.citations.urls is typically a list of objects with .title and .url
                citations = [{"title": u.title or u.url, "url": u.url} for u in citations_attr.urls]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")

    if not answer_text.strip():
        raise HTTPException(status_code=502, detail="No answer generated.")

    return AskResponse(
        answer=answer_text.strip(),
        reasoning=reasoning_text,
        citations=[Citation(**c) for c in citations]
    )

@app.get("/stream")
async def stream(query: str, show_full_reasoning: bool = True, x_openai_key: Optional[str] = Header(default=None)):
    """
    Server-Sent Events (SSE) stream.
    Events: 'reasoning', 'answer', 'citations', 'done', 'error'
    """

    api_key = get_effective_api_key(x_openai_key)
    await ensure_initialized(api_key)

    if not STATE.agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            # Run synchronously but yield as events
            for chunk in STATE.agent.run(
                query,
                stream=True,
                show_full_reasoning=show_full_reasoning,
                stream_intermediate_steps=True,
            ):
                rc = getattr(chunk, "reasoning_content", None)
                if rc:
                    yield f"event: reasoning\ndata: {rc}\n\n"

                content = getattr(chunk, "content", None)
                if content and getattr(chunk, "event", None) in {RunEvent.run_response, RunEvent.run_completed}:
                    if isinstance(content, str):
                        yield f"event: answer\ndata: {content}\n\n"

                citations_attr = getattr(chunk, "citations", None)
                if citations_attr and getattr(citations_attr, "urls", None):
                    # Send once; client can dedupe if multiple
                    payload = [{"title": (u.title or u.url), "url": u.url} for u in citations_attr.urls]
                    yield f"event: citations\ndata: {payload}\n\n"

            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Optional: run with `python app.py` (or use `uvicorn app:app --reload`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
