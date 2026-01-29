"""
Graph RAG FastAPI Backend

Provides REST API endpoints for querying the knowledge graph.
Supports both local and global search modes.
"""

import re
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from src.community_detection import load_communities
from src.graph_builder import KnowledgeGraph
from src.graph_retriever import GraphRetriever, create_retriever
from src.langgraph_engine import LangGraphEngine, create_langgraph_engine
from src.utils.graph_utils import Community


# Load config
def load_config() -> DictConfig:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    return OmegaConf.load(config_path)


# Initialize config
cfg = load_config()

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format=cfg.LOGGING.format,
    level=cfg.LOGGING.level,
)


class AppState:
    """Application state container for dependency injection."""

    kg: KnowledgeGraph | None = None
    communities: dict[str, Community] | None = None
    retriever: GraphRetriever | None = None
    llm: OllamaLLM | ChatAnthropic | None = None
    langgraph_engine: LangGraphEngine | None = None


# Application state instance
state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Loading Graph RAG components...")

    graph_db_dir = Path(cfg.PATHS.graph_db_dir)

    # Load knowledge graph
    try:
        state.kg = KnowledgeGraph.load(graph_db_dir)
        logger.info(
            f"Loaded knowledge graph: {state.kg.num_nodes} nodes, {state.kg.num_edges} edges"
        )
    except Exception as e:
        logger.error(f"Failed to load knowledge graph: {e}")
        state.kg = KnowledgeGraph()

    # Load communities
    communities_path = graph_db_dir / "community_summaries.json"
    if communities_path.exists():
        try:
            state.communities = load_communities(communities_path)
            logger.info(f"Loaded {len(state.communities)} communities")
        except Exception as e:
            logger.error(f"Failed to load communities: {e}")
            state.communities = {}
    else:
        state.communities = {}

    # Initialize retriever
    state.retriever = create_retriever(state.kg, state.communities, cfg)

    # Initialize LLM based on provider
    provider = cfg.GENERATION.get("provider", "ollama")
    if provider == "anthropic":
        state.llm = ChatAnthropic(
            model=cfg.GENERATION.model,
            temperature=cfg.GENERATION.temperature,
            max_tokens=cfg.GENERATION.max_tokens,
        )
        logger.info(f"Using Anthropic model: {cfg.GENERATION.model}")
    else:
        state.llm = OllamaLLM(
            model=cfg.GENERATION.model,
            temperature=cfg.GENERATION.temperature,
            num_predict=cfg.GENERATION.max_tokens,
        )
        logger.info(f"Using Ollama model: {cfg.GENERATION.model}")

    # Initialize LangGraph engine with conversation memory
    state.langgraph_engine = create_langgraph_engine(
        retriever=state.retriever,
        llm=state.llm,
        checkpointer_type="memory",  # Use "sqlite" for persistence
    )

    logger.info("Graph RAG backend ready with LangGraph conversation memory!")

    yield  # Application runs here

    # Shutdown (cleanup if needed)
    logger.info("Shutting down Graph RAG backend...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Graph RAG API",
    description="Knowledge Graph-based Retrieval Augmented Generation API",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware with configurable origins
cors_origins = list(cfg.SERVER.get("cors_origins", ["*"]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="The question to answer")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation continuity (null for new session)"
    )
    temperature: Optional[float] = Field(
        default=None, description="LLM temperature (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens in response"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    session_id: str = Field(description="Session ID for conversation continuity")
    resolved_mode: str = Field(description="Search mode used (always combined)")
    context_summary: dict
    message_count: int = Field(description="Total messages in conversation")
    chain_of_thought: Optional[str] = None
    prompt: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    graph_nodes: int
    graph_edges: int
    num_communities: int
    llm_model: str


class GraphStatsResponse(BaseModel):
    """Response model for graph stats endpoint."""

    num_nodes: int
    num_edges: int
    num_communities: int
    entity_types: dict
    relationship_types: dict


# Prompt template
GRAPH_RAG_PROMPT = """You are an expert assistant specializing in Singapore government policies, schemes, and regulations.

You have access to a knowledge graph containing information about:
- Government organizations (CPF Board, HDB, IRAS, MOH, etc.)
- Schemes and programs (CPF LIFE, MediShield Life, BTO, grants, etc.)
- Eligibility requirements, monetary amounts, and processes
- Relationships between different policies and entities

CONTEXT FROM KNOWLEDGE GRAPH:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context above
2. If the context contains relevant information, provide a complete and accurate answer
3. Reference specific entities, amounts, or requirements from the context when applicable
4. If the context doesn't contain enough information, clearly state what's missing
5. Do not make up information that isn't in the context
6. Be specific - include exact figures, eligibility criteria, and requirements when available

ANSWER:"""


def extract_chain_of_thought(text: str) -> tuple[str, str]:
    """
    Remove <think>...</think> tags and extract chain of thought.

    Args:
        text: LLM output potentially containing think tags

    Returns:
        Tuple of (chain_of_thought, final_answer)
    """
    chain = ""
    final = text

    if "<think>" in text and "</think>" in text:
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            chain = match.group(1).strip()
            final = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return chain, final


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and return system status."""
    return HealthResponse(
        status="ok" if state.kg is not None else "degraded",
        graph_nodes=state.kg.num_nodes if state.kg else 0,
        graph_edges=state.kg.num_edges if state.kg else 0,
        num_communities=len(state.communities) if state.communities else 0,
        llm_model=cfg.GENERATION.model,
    )


@app.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Get knowledge graph statistics."""
    if state.kg is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")

    stats = state.kg.get_statistics()
    return GraphStatsResponse(
        num_nodes=stats["num_nodes"],
        num_edges=stats["num_edges"],
        num_communities=len(state.communities) if state.communities else 0,
        entity_types=stats["entity_types"],
        relationship_types=stats["relationship_types"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge graph and generate an answer with conversation memory."""
    if state.langgraph_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Use LangGraph engine for conversational query
    try:
        result = state.langgraph_engine.query(
            question=question,
            session_id=request.session_id,
            temperature=request.temperature or cfg.GENERATION.temperature,
            max_tokens=request.max_tokens or cfg.GENERATION.max_tokens,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Extract chain of thought if present in answer
    answer = result["answer"]
    chain_of_thought = None
    if "<think>" in answer and "</think>" in answer:
        chain_of_thought, answer = extract_chain_of_thought(answer)

    return QueryResponse(
        answer=answer,
        session_id=result["session_id"],
        resolved_mode=result.get("resolved_mode", "combined"),
        context_summary=result.get("context_summary", {}),
        message_count=result.get("message_count", 0),
        chain_of_thought=chain_of_thought,
        prompt=result.get("context", ""),
    )


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Get conversation history for a session."""
    if state.langgraph_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    history = state.langgraph_engine.get_conversation_history(session_id)

    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history),
    }


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session (start fresh)."""
    if state.langgraph_engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    state.langgraph_engine.clear_conversation(session_id)

    return {
        "session_id": session_id,
        "status": "cleared",
        "message": "Start a new conversation with the same or new session_id",
    }


@app.get("/entities")
async def list_entities(limit: int = 100, entity_type: str | None = None):
    """List entities in the knowledge graph."""
    if state.kg is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")

    entities = []
    for node_id, data in state.kg.get_all_nodes():
        if entity_type and data.get("type", "").lower() != entity_type.lower():
            continue

        entities.append(
            {
                "id": node_id,
                "name": data.get("name", node_id),
                "type": data.get("type", "Unknown"),
                "description": data.get("description", "")[:200],
                "degree": state.kg.graph.degree(node_id),
            }
        )

        if len(entities) >= limit:
            break

    # Sort by degree
    entities.sort(key=lambda x: x["degree"], reverse=True)
    return {"entities": entities, "total": state.kg.num_nodes}


@app.get("/communities")
async def list_communities():
    """List all communities."""
    if not state.communities:
        return {"communities": [], "total": 0}

    comm_list = []
    for comm_id, comm in state.communities.items():
        comm_list.append(
            {
                "id": comm_id,
                "level": comm.level,
                "title": comm.title,
                "num_entities": len(comm.entity_ids),
                "summary": (
                    comm.summary[:300] + "..."
                    if len(comm.summary) > 300
                    else comm.summary
                ),
            }
        )

    return {"communities": comm_list, "total": len(state.communities)}


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.app_backend:app",
        host=cfg.SERVER.host,
        port=cfg.SERVER.backend_port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
