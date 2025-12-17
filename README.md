# Graph RAG

A knowledge graph-based Retrieval Augmented Generation (RAG) system that extracts entities and relationships from documents, builds a knowledge graph, detects communities, and provides intelligent question-answering capabilities.

This implementation is designed for **Singapore Government Policies** - specifically CPF (Central Provident Fund), HDB housing, healthcare schemes, and related policies where cross-referencing and multi-hop reasoning are essential.

## Overview

Traditional RAG systems retrieve relevant text chunks using vector similarity. Graph RAG enhances this by:

1. **Extracting structured knowledge** - Uses LLM to identify entities and relationships from documents
2. **Building a knowledge graph** - Creates a NetworkX graph with entities as nodes and relationships as edges
3. **Detecting communities** - Groups related entities using Leiden/Louvain algorithms
4. **Dual retrieval strategies**:
   - **Local Search**: Entity-based graph traversal for specific queries
   - **Global Search**: Community-based search for broad/summary queries

## Architecture

```
Documents -> Entity Extraction -> Knowledge Graph -> Community Detection
                                        |
                                Community Summaries
                                        |
Query -> Local/Global Search -> Graph Context -> LLM -> Answer
```

## Features

- LLM-based entity and relationship extraction
- NetworkX knowledge graph with JSON persistence
- Hierarchical community detection (Leiden algorithm)
- Automatic community summarization
- Dual retrieval modes (local + global)
- FastAPI REST API
- Gradio web interface
- Extraction result caching

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) with models:
  - LLM model (e.g., `qwen2.5:14b`, `llama3.2:3b`, or `deepseek-r1:7b`)
  - Embedding model: `nomic-embed-text`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TerrenceYeoh/GraphRAG.git
   cd GraphRAG
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Install Ollama models**
   ```bash
   ollama pull qwen2.5:14b       # or your preferred LLM
   ollama pull nomic-embed-text  # for embeddings
   ```

4. **Add your documents**
   ```bash
   mkdir -p corpus/raw
   # Copy your documents (PDF, JSON, TXT, CSV, MD) to corpus/raw/
   ```

## Usage

### Quick Start

```bash
# Build the knowledge graph index
python main.py build

# Start the API server and web UI
python main.py serve
```

### Individual Commands

```bash
python main.py build      # Build/rebuild the graph index
python main.py backend    # Start FastAPI backend (port 8001)
python main.py frontend   # Start Gradio UI (port 7861)
python main.py serve      # Start both backend and frontend
```

### Using uv directly

```bash
uv run python -m src.build_graph_index   # Build index
uv run python -m src.app_backend         # Start backend
uv run python -m src.gradio_frontend     # Start frontend
```

## Configuration

Edit `conf/config.yaml` to customize:

```yaml
# Models
EXTRACTION:
  model: "qwen2.5:14b"           # LLM for entity extraction
  embedding_model: "nomic-embed-text"
  entity_types:
    - Organization      # CPF Board, HDB, IRAS, ministries
    - Scheme            # CPF LIFE, MediShield Life, ComCare
    - Grant             # EHG, Family Grant, PSG
    - Requirement       # Eligibility criteria, conditions
    - MonetaryAmount    # $80,000, 17%, income ceilings
    - AgeGroup          # 55 and above, seniors, first-timers
    - Account           # OA, SA, MA, RA
    - Property          # BTO, resale flat, HDB, private

# Document Processing
DOCUMENT:
  chunk_size: 1500
  chunk_overlap: 200

# Community Detection
COMMUNITY:
  algorithm: "leiden"
  min_community_size: 3

# Retrieval
RETRIEVAL:
  local_search_hops: 2           # Graph traversal depth
  global_top_communities: 8      # Communities for global search

# Server
SERVER:
  backend_port: 8001
  frontend_port: 7861
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Graph Statistics
```bash
GET /stats
```

### Query
```bash
POST /query
Content-Type: application/json

{
    "question": "How much CPF can I use for my HDB downpayment?",
    "mode": "auto",           # auto, local, or global
    "temperature": 0.5,
    "max_tokens": 1024
}
```

### List Entities
```bash
GET /entities?limit=100&entity_type=Organization
```

### List Communities
```bash
GET /communities
```

## Project Structure

```
GraphRAG/
├── conf/
│   └── config.yaml              # Configuration
├── src/
│   ├── entity_extraction.py     # LLM entity/relation extraction
│   ├── graph_builder.py         # Knowledge graph construction
│   ├── community_detection.py   # Leiden/Louvain clustering
│   ├── community_summarizer.py  # Community summaries
│   ├── graph_retriever.py       # Local/Global search
│   ├── build_graph_index.py     # Indexing pipeline
│   ├── app_backend.py           # FastAPI server
│   ├── gradio_frontend.py       # Web UI
│   └── utils/
│       └── graph_utils.py       # Data structures
├── graph_db/                    # Generated graph data
├── chroma_db/                   # Community embeddings
├── corpus/
│   └── raw/                     # Your documents
├── main.py                      # CLI entry point
├── pyproject.toml               # Dependencies
├── PLAN.md                      # Implementation notes
└── README.md                    # This file
```

## How It Works

### 1. Entity Extraction
The system uses an LLM to extract entities and relationships from each document chunk:

```json
{
    "entities": [
        {"name": "CPF LIFE", "type": "Scheme", "description": "Lifelong income scheme for retirement"},
        {"name": "Ordinary Account", "type": "Account", "description": "CPF account for housing, education, insurance"}
    ],
    "relationships": [
        {"source": "CPF Board", "target": "CPF LIFE", "relation_type": "ADMINISTERS"},
        {"source": "Ordinary Account", "target": "HDB", "relation_type": "CAN_BE_USED_FOR"}
    ]
}
```

### 2. Knowledge Graph
Entities become nodes, relationships become edges:

```
(CPF Board) --[ADMINISTERS]--> (CPF LIFE)
(Ordinary Account) --[CAN_BE_USED_FOR]--> (HDB Downpayment)
(Enhanced Housing Grant) --[REQUIRES]--> (Income Ceiling $9,000)
```

### 3. Community Detection
Related entities are grouped into communities using the Leiden algorithm. Each community gets an LLM-generated summary.

### 4. Query Processing

**Local Search** (for specific queries):
1. Extract entities from question
2. Find matching nodes in graph
3. Traverse neighborhood (1-2 hops)
4. Build context from subgraph

**Global Search** (for broad queries):
1. Search community summaries by similarity
2. Retrieve top-k relevant communities
3. Use summaries as context

## Example Output

Query: "What is the income ceiling for Enhanced Housing Grant?"

```json
{
    "answer": "The income ceiling for the Enhanced CPF Housing Grant (EHG) is $9,000 per month for households. First-timer families with income not exceeding this ceiling can receive up to $80,000 in grants...",
    "search_mode": "local",
    "context_summary": {
        "matched_entities": ["Enhanced Housing Grant", "Income Ceiling", "First-Timer"],
        "total_nodes": 12,
        "total_edges": 18
    }
}
```

## Performance Tips

- Use a larger LLM model (e.g., `qwen2.5:14b`) for better entity extraction
- Extraction results are cached in `graph_db/extraction_cache/`
- Rebuilds are fast when using cache
- For large corpora, consider batching and parallel processing

## Troubleshooting

### No communities detected
- Graph may be too sparse (many isolated nodes)
- Try lowering `min_community_size` in config
- Use a larger LLM for better relationship extraction

### Slow extraction
- Entity extraction runs ~1-2s per chunk
- Use cache by running build again (instant rebuild)
- Consider a smaller/faster model

### Port already in use
- Change `backend_port` and `frontend_port` in config.yaml

## License

MIT License

## Acknowledgments

- Based on the [Microsoft GraphRAG](https://github.com/microsoft/graphrag) architecture
- Uses [Ollama](https://ollama.ai/) for local LLM inference
- Built with [LangChain](https://langchain.com/), [NetworkX](https://networkx.org/), and [FastAPI](https://fastapi.tiangolo.com/)
