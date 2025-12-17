"""
Graph RAG Indexing Pipeline

Main script to build the knowledge graph index from documents:
1. Load and chunk documents
2. Extract entities and relationships using LLM
3. Build knowledge graph
4. Detect communities
5. Generate community summaries
6. Create vector store for community embeddings
"""

import hashlib
import sys
from pathlib import Path

import hydra
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from omegaconf import DictConfig
import fitz  # PyMuPDF
import pandas as pd
import json

from src.entity_extraction import EntityExtractor, create_extractor
from src.graph_builder import KnowledgeGraph, build_graph_from_extractions
from src.community_detection import CommunityDetector, create_detector, save_communities
from src.community_summarizer import CommunitySummarizer, create_summarizer, save_community_summaries
from src.utils.graph_utils import save_json


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging."""
    log_dir = Path(cfg.PATHS.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format=cfg.LOGGING.format,
        level=cfg.LOGGING.level,
    )
    logger.add(
        log_dir / "build_graph_index.log",
        format=cfg.LOGGING.format,
        level="DEBUG",
        rotation="10 MB",
    )


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF file."""
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {path}: {e}")
        return ""


def extract_text_from_json(path: Path) -> str:
    """Extract text from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            # FAQ format
            if "faq_items" in data:
                texts = []
                for item in data["faq_items"]:
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    texts.append(f"Q: {q}\nA: {a}")
                return "\n\n".join(texts)

            # Webpage format
            if "text_chunks" in data:
                return "\n\n".join(data["text_chunks"])

            # Generic: extract all string values
            return " ".join(str(v) for v in data.values() if isinstance(v, str))

        elif isinstance(data, list):
            return "\n\n".join(str(item) for item in data)

        return str(data)

    except Exception as e:
        logger.error(f"Failed to extract text from JSON {path}: {e}")
        return ""


def extract_text_from_csv(path: Path) -> str:
    """Extract text from CSV file."""
    try:
        df = pd.read_csv(path)
        return df.to_string()
    except Exception as e:
        logger.error(f"Failed to extract text from CSV {path}: {e}")
        return ""


def extract_text_from_txt(path: Path) -> str:
    """Extract text from TXT file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to extract text from TXT {path}: {e}")
        return ""


def load_documents(corpus_dir: Path, extensions: list[str]) -> list[tuple[str, str, str]]:
    """
    Load documents from corpus directory.

    Returns:
        List of (doc_id, source_path, text) tuples
    """
    documents = []

    for ext in extensions:
        for file_path in corpus_dir.rglob(f"*{ext}"):
            # Extract text based on file type
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == ".json":
                text = extract_text_from_json(file_path)
            elif ext == ".csv":
                text = extract_text_from_csv(file_path)
            elif ext in [".txt", ".md"]:
                text = extract_text_from_txt(file_path)
            else:
                continue

            if text.strip():
                doc_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
                documents.append((doc_id, str(file_path), text))
                logger.debug(f"Loaded {file_path}: {len(text)} chars")

    logger.info(f"Loaded {len(documents)} documents from {corpus_dir}")
    return documents


def chunk_documents(
    documents: list[tuple[str, str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str]]:
    """
    Split documents into chunks.

    Returns:
        List of (chunk_id, text) tuples
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for doc_id, source_path, text in documents:
        doc_chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(doc_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunks.append((chunk_id, chunk_text))

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main indexing pipeline."""
    setup_logging(cfg)
    logger.info("Starting Graph RAG indexing pipeline")

    # Create output directories
    graph_db_dir = Path(cfg.PATHS.graph_db_dir)
    graph_db_dir.mkdir(parents=True, exist_ok=True)

    vector_db_dir = Path(cfg.PATHS.vector_db_dir)
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load documents
    logger.info("Step 1: Loading documents...")
    corpus_dir = Path(cfg.PATHS.corpus_dir)

    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        logger.info("Please copy your documents to the corpus directory and try again.")
        return

    documents = load_documents(
        corpus_dir,
        list(cfg.DOCUMENT.supported_extensions),
    )

    if not documents:
        logger.error("No documents found in corpus directory")
        return

    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunks = chunk_documents(
        documents,
        cfg.DOCUMENT.chunk_size,
        cfg.DOCUMENT.chunk_overlap,
    )

    # Step 3: Extract entities and relationships
    logger.info("Step 3: Extracting entities and relationships...")
    extractor = create_extractor(cfg)
    extraction_results = extractor.extract_from_chunks(chunks, use_cache=True)

    # Merge all extractions
    entities, relationships = extractor.merge_results(extraction_results)
    logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")

    # Save extraction results
    save_json(
        [e.to_dict() for e in entities],
        graph_db_dir / "entities.json",
    )
    save_json(
        [r.to_dict() for r in relationships],
        graph_db_dir / "relationships.json",
    )

    # Step 4: Build knowledge graph
    logger.info("Step 4: Building knowledge graph...")
    kg = build_graph_from_extractions(entities, relationships)
    kg.save(graph_db_dir)

    # Log graph statistics
    stats = kg.get_statistics()
    logger.info(f"Graph statistics: {stats}")

    # Step 5: Detect communities
    logger.info("Step 5: Detecting communities...")
    detector = create_detector(cfg)
    communities = detector.detect_communities(kg)
    save_communities(communities, graph_db_dir / "communities.json")
    logger.info(f"Detected {len(communities)} communities")

    # Step 6: Generate community summaries
    logger.info("Step 6: Generating community summaries...")
    summarizer = create_summarizer(cfg)
    communities = summarizer.summarize_all_communities(communities, kg)
    save_community_summaries(communities, graph_db_dir / "community_summaries.json")

    # Step 7: Create community vector store
    logger.info("Step 7: Creating community vector store...")
    from src.graph_retriever import create_retriever

    retriever = create_retriever(kg, communities, cfg)
    logger.info("Community vector store created")

    # Summary
    logger.info("=" * 50)
    logger.info("Graph RAG indexing complete!")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Entities: {len(entities)}")
    logger.info(f"  Relationships: {len(relationships)}")
    logger.info(f"  Graph nodes: {kg.num_nodes}")
    logger.info(f"  Graph edges: {kg.num_edges}")
    logger.info(f"  Communities: {len(communities)}")
    logger.info(f"  Output directory: {graph_db_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
