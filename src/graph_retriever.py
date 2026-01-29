"""
Graph Retriever Module

Implements combined search for Graph RAG, merging:
- Local Search: Entity-based graph traversal with source chunk retrieval
- Global Search: Community summary vector similarity search
"""

from pathlib import Path

import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from src.graph_builder import KnowledgeGraph
from src.utils.graph_utils import (Community, load_chunk_texts,
                                   normalize_entity_name, strip_think_tags)

# Prompt for extracting entities from user query
QUERY_ENTITY_EXTRACTION_PROMPT = """/no_think
Extract the key entities, topics, and concepts from this question.

QUESTION: {question}

Extract ALL relevant terms including:
- Named entities (organizations, schemes, programs)
- Key concepts (e.g., "grants", "housing", "retirement")
- Descriptive terms (e.g., "first time buyer", "eligible")

List each entity/concept on a separate line. Be comprehensive - extract 3-6 terms.
Do NOT respond with "NONE" - always extract at least the main topic words.

ENTITIES:"""


class GraphRetriever:
    """
    Retriever that uses knowledge graph for context retrieval.
    Uses combined search (local + global) for comprehensive context.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        communities: dict[str, Community],
        llm_model: str = "deepseek-r1:7b",
        embedding_model: str = "nomic-embed-text",
        vector_db_path: Path | None = None,
        extraction_cache_dir: Path | None = None,
        local_search_hops: int = 2,
        local_max_entities: int = 25,
        local_max_edges: int = 50,
        local_max_source_chunks: int = 5,
        global_top_communities: int = 8,
    ):
        """
        Initialize the graph retriever.

        Args:
            kg: Knowledge graph
            communities: Dictionary of communities with summaries
            llm_model: Model for query analysis
            embedding_model: Model for embeddings
            vector_db_path: Path to vector store for community embeddings
            extraction_cache_dir: Path to extraction cache for source chunk retrieval
            local_search_hops: Number of hops for local search
            local_max_entities: Maximum entities to return in local search
            local_max_edges: Maximum edges to include in context
            local_max_source_chunks: Maximum source chunks to include in context
            global_top_communities: Number of communities for global search
        """
        self.kg = kg
        self.communities = communities
        self.local_search_hops = local_search_hops
        self.local_max_entities = local_max_entities
        self.local_max_edges = local_max_edges
        self.local_max_source_chunks = local_max_source_chunks
        self.global_top_communities = global_top_communities
        self.extraction_cache_dir = extraction_cache_dir

        # Store LLM model name for query analysis
        self.llm_model = llm_model

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Initialize or load community vector store
        self.vector_db_path = vector_db_path
        self.community_vectordb = None

        if vector_db_path and communities:
            self._initialize_community_vectordb()

        logger.info(
            f"Initialized GraphRetriever with {kg.num_nodes} nodes, "
            f"{len(communities)} communities"
        )

    def _initialize_community_vectordb(self) -> None:
        """Initialize vector store for community summaries."""
        if not self.communities:
            return

        # Check if vector store exists
        if self.vector_db_path and (self.vector_db_path / "chroma.sqlite3").exists():
            try:
                self.community_vectordb = Chroma(
                    persist_directory=str(self.vector_db_path),
                    embedding_function=self.embeddings,
                    collection_name="community_summaries",
                )
                logger.info("Loaded existing community vector store")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")

        # Create new vector store from community summaries
        texts = []
        metadatas = []
        ids = []

        for comm_id, community in self.communities.items():
            if community.summary:
                texts.append(community.summary)
                metadatas.append(
                    {
                        "community_id": comm_id,
                        "level": community.level,
                        "title": community.title,
                        "num_entities": len(community.entity_ids),
                    }
                )
                ids.append(comm_id)

        if texts:
            self.community_vectordb = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=(
                    str(self.vector_db_path) if self.vector_db_path else None
                ),
                collection_name="community_summaries",
            )
            logger.info(f"Created community vector store with {len(texts)} summaries")

    def extract_query_entities(self, question: str) -> list[str]:
        """
        Extract entity names mentioned in the query.

        Args:
            question: User question

        Returns:
            List of entity names
        """
        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(question=question)

        try:
            result = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={"temperature": 0.0},
                think=False,  # Disable thinking mode for speed
            )
            response = result["response"]
            response = strip_think_tags(response)

            # Parse entity names
            entities = []
            for line in response.split("\n"):
                line = line.strip()
                # Remove common prefixes
                for prefix in ["-", "*", "•", "1.", "2.", "3."]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                if line and len(line) > 1:
                    entities.append(line)

            return entities
        except Exception as e:
            logger.warning(f"Failed to extract query entities: {e}")
            return []

    def local_search(self, question: str) -> dict:
        """
        Perform local search using entity-based graph traversal.

        Args:
            question: User question

        Returns:
            Dictionary with retrieved context and metadata
        """
        # Extract entities from query
        query_entities = self.extract_query_entities(question)
        logger.debug(f"Extracted query entities: {query_entities}")

        # Find matching nodes in graph
        matched_nodes = []
        for entity_name in query_entities:
            normalized = normalize_entity_name(entity_name)

            # Exact match
            if normalized in self.kg.graph:
                matched_nodes.append(normalized)
                continue

            # Word-based matching: split entity into words and find nodes containing those words
            entity_words = set(normalized.replace("_", " ").split())
            # Filter out very short/common words
            entity_words = {w for w in entity_words if len(w) > 2}

            best_matches = []
            for node_id in self.kg.graph.nodes():
                node_words = set(node_id.replace("_", " ").split())
                # Count how many entity words appear in the node
                overlap = entity_words & node_words
                if overlap:
                    # Score by number of matching words and node specificity
                    score = len(overlap) / max(len(node_words), 1)
                    best_matches.append((node_id, len(overlap), score))

            # Sort by overlap count (desc), then by score (desc)
            best_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Add top matches (nodes with at least 1 significant word match)
            for node_id, overlap_count, _ in best_matches[:5]:
                if node_id not in matched_nodes:
                    matched_nodes.append(node_id)

        if not matched_nodes:
            # Fallback: use most connected nodes
            degrees = self.kg.get_node_degrees()
            matched_nodes = sorted(
                degrees.keys(), key=lambda x: degrees[x], reverse=True
            )[: self.local_max_entities // 2]
            logger.debug("No direct matches, using top connected nodes")

        # Get subgraph around matched nodes
        all_relevant_nodes = set(matched_nodes)
        for node in matched_nodes:
            neighbors = self.kg.get_neighbors(node, hops=self.local_search_hops)
            all_relevant_nodes.update(neighbors[: self.local_max_entities])

        # Limit total nodes efficiently
        if len(all_relevant_nodes) > self.local_max_entities:
            # Prioritize matched nodes and their direct neighbors
            # Use set for O(1) membership check in sort key
            matched_set = set(matched_nodes)
            # Compute degrees once and sort
            nodes_with_priority = [
                (node, node in matched_set, self.kg.graph.degree(node))
                for node in all_relevant_nodes
            ]
            # Sort by (is_matched, degree) descending and take top N
            nodes_with_priority.sort(key=lambda x: (x[1], x[2]), reverse=True)
            all_relevant_nodes = {n[0] for n in nodes_with_priority[: self.local_max_entities]}

        # Build context from subgraph
        subgraph = self.kg.get_subgraph(list(all_relevant_nodes), hops=0)

        # Collect source chunk IDs from relevant nodes (prioritize matched nodes)
        source_chunk_ids = []
        seen_chunks = set()
        # First add chunks from matched nodes
        for node_id in matched_nodes:
            node_data = self.kg.get_node_data(node_id)
            if node_data:
                chunk_id = node_data.get("source_chunk", "")
                if chunk_id and chunk_id not in seen_chunks:
                    source_chunk_ids.append(chunk_id)
                    seen_chunks.add(chunk_id)
        # Then add chunks from other relevant nodes
        for node_id in all_relevant_nodes:
            if node_id in matched_nodes:
                continue
            node_data = self.kg.get_node_data(node_id)
            if node_data:
                chunk_id = node_data.get("source_chunk", "")
                if chunk_id and chunk_id not in seen_chunks:
                    source_chunk_ids.append(chunk_id)
                    seen_chunks.add(chunk_id)
            if len(source_chunk_ids) >= self.local_max_source_chunks:
                break

        # Load source texts from extraction cache
        source_context = ""
        source_chunks_loaded = 0
        if self.extraction_cache_dir and source_chunk_ids:
            chunk_texts = load_chunk_texts(
                source_chunk_ids[: self.local_max_source_chunks],
                self.extraction_cache_dir,
            )
            if chunk_texts:
                source_context = "=== SOURCE DOCUMENTS ===\n"
                for chunk_id, text in chunk_texts.items():
                    source_context += f"\n[Chunk {chunk_id}]\n{text}\n"
                source_chunks_loaded = len(chunk_texts)
                logger.debug(f"Loaded {source_chunks_loaded} source chunks")

        # Get community context for matched nodes
        community_context = ""
        for node_id in matched_nodes[:3]:  # Top 3 matched nodes
            for comm in self.communities.values():
                if node_id in comm.entity_ids and comm.summary:
                    community_context += (
                        f"\n[Community: {comm.title}]\n{comm.summary[:300]}...\n"
                    )
                    break

        context = subgraph.to_context_string(
            max_nodes=self.local_max_entities, max_edges=self.local_max_edges
        )

        # Assemble final context: source docs first, then community, then graph
        if source_context:
            context = f"{source_context}\n{context}"
        if community_context:
            context = f"{context}\n\n=== COMMUNITY CONTEXT ==={community_context}"

        return {
            "context": context,
            "mode": "local",
            "matched_entities": query_entities,
            "matched_nodes": list(matched_nodes),
            "total_nodes": len(all_relevant_nodes),
            "total_edges": subgraph.num_edges,
            "source_chunks_loaded": source_chunks_loaded,
        }

    def combined_search(self, question: str) -> dict:
        """
        Perform combined local + global search for comprehensive context.

        Runs both local (entity-based) and global (community-based) search,
        then merges the results into a single context.

        Args:
            question: User question

        Returns:
            Dictionary with merged context and metadata from both searches
        """
        # Run both searches
        local_result = self.local_search(question)
        global_result = self._global_search_raw(question)

        # Merge contexts: local context first, then global community summaries
        context_parts = []

        # Local context (includes source docs, entity graph, community context)
        local_context = local_result.get("context", "")
        if local_context:
            context_parts.append(local_context)

        # Global context (community summaries from vector search)
        global_context = global_result.get("context", "")
        if global_context:
            context_parts.append(
                f"\n=== GLOBAL COMMUNITY SUMMARIES ===\n{global_context}"
            )

        return {
            "context": "\n".join(context_parts),
            "mode": "combined",
            "matched_entities": local_result.get("matched_entities", []),
            "matched_nodes": local_result.get("matched_nodes", []),
            "total_nodes": local_result.get("total_nodes", 0),
            "total_edges": local_result.get("total_edges", 0),
            "source_chunks_loaded": local_result.get("source_chunks_loaded", 0),
            "communities_searched": global_result.get("communities_searched", []),
            "total_communities": global_result.get("total_communities", 0),
        }

    def _global_search_raw(self, question: str) -> dict:
        """
        Perform raw global search without fallback to local.

        Used internally by combined_search and global_search.

        Args:
            question: User question

        Returns:
            Dictionary with retrieved context and metadata,
            or empty context if vector store unavailable
        """
        if not self.community_vectordb:
            logger.warning("Community vector store not initialized")
            return {
                "context": "",
                "mode": "global",
                "communities_searched": [],
                "total_communities": 0,
                "total_entities": 0,
            }

        try:
            results = self.community_vectordb.similarity_search_with_score(
                question, k=self.global_top_communities
            )
        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return {
                "context": "",
                "mode": "global",
                "communities_searched": [],
                "total_communities": 0,
                "total_entities": 0,
            }

        # Build context from community summaries
        context_parts = []
        community_ids = []
        total_entities = 0

        for doc, score in results:
            comm_id = doc.metadata.get("community_id", "unknown")
            title = doc.metadata.get("title", "Unknown Community")
            level = doc.metadata.get("level", 0)
            num_entities = doc.metadata.get("num_entities", 0)

            context_parts.append(
                f"=== {title} (Level {level}, {num_entities} entities) ===\n"
                f"{doc.page_content}\n"
            )
            community_ids.append(comm_id)
            total_entities += num_entities

        context = "\n".join(context_parts)

        return {
            "context": context,
            "mode": "global",
            "communities_searched": community_ids,
            "total_communities": len(results),
            "total_entities": total_entities,
        }

    def global_search(self, question: str) -> dict:
        """
        Perform global search using community summaries.

        Falls back to local search if community vector store is unavailable.

        Args:
            question: User question

        Returns:
            Dictionary with retrieved context and metadata
        """
        result = self._global_search_raw(question)
        if not result["context"]:
            logger.warning("Global search returned no results, falling back to local")
            return self.local_search(question)
        return result

def create_retriever(
    kg: KnowledgeGraph,
    communities: dict[str, Community],
    cfg,
) -> GraphRetriever:
    """Create a GraphRetriever from config."""
    vector_db_path = Path(cfg.PATHS.vector_db_dir) / "communities"
    extraction_cache_dir = Path(cfg.PATHS.graph_db_dir) / "extraction_cache"

    return GraphRetriever(
        kg=kg,
        communities=communities,
        llm_model=cfg.EXTRACTION.model,
        embedding_model=cfg.EXTRACTION.embedding_model,
        vector_db_path=vector_db_path,
        extraction_cache_dir=extraction_cache_dir,
        local_search_hops=cfg.RETRIEVAL.local_search_hops,
        local_max_entities=cfg.RETRIEVAL.local_max_entities,
        local_max_edges=cfg.RETRIEVAL.get("local_max_edges", 50),
        local_max_source_chunks=cfg.RETRIEVAL.get("local_max_source_chunks", 5),
        global_top_communities=cfg.RETRIEVAL.global_top_communities,
    )
