"""
Graph Retriever Module

Implements local and global search strategies for Graph RAG:
- Local Search: Entity-based traversal for specific queries
- Global Search: Community-based search for broad/summary queries
"""

from pathlib import Path
from typing import Literal

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from loguru import logger

from src.graph_builder import KnowledgeGraph
from src.utils.graph_utils import (Community, normalize_entity_name,
                                   strip_think_tags)

# Prompt for extracting entities from user query
QUERY_ENTITY_EXTRACTION_PROMPT = """Extract the key entities mentioned in this question.

QUESTION: {question}

List ONLY the entity names mentioned or implied in the question, one per line.
If no specific entities are mentioned, respond with "NONE".

ENTITIES:"""


# Prompt for determining search mode
SEARCH_MODE_PROMPT = """Analyze this question and determine the best search strategy.

QUESTION: {question}

If the question:
- Asks about specific entities, people, programs, or facts → respond with "LOCAL"
- Asks for summaries, overviews, comparisons, or general information → respond with "GLOBAL"
- Is unclear or could go either way → respond with "LOCAL"

Respond with only one word: LOCAL or GLOBAL"""


class GraphRetriever:
    """
    Retriever that uses knowledge graph for context retrieval.
    Supports both local (entity-based) and global (community-based) search.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        communities: dict[str, Community],
        llm_model: str = "deepseek-r1:7b",
        embedding_model: str = "nomic-embed-text",
        vector_db_path: Path | None = None,
        local_search_hops: int = 2,
        local_max_entities: int = 25,
        local_max_edges: int = 50,
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
            local_search_hops: Number of hops for local search
            local_max_entities: Maximum entities to return in local search
            local_max_edges: Maximum edges to include in context
            global_top_communities: Number of communities for global search
        """
        self.kg = kg
        self.communities = communities
        self.local_search_hops = local_search_hops
        self.local_max_entities = local_max_entities
        self.local_max_edges = local_max_edges
        self.global_top_communities = global_top_communities

        # Initialize LLM for query analysis
        self.llm = OllamaLLM(model=llm_model, temperature=0.0)

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

    def determine_search_mode(self, question: str) -> Literal["local", "global"]:
        """
        Determine whether to use local or global search.

        Args:
            question: User question

        Returns:
            "local" or "global"
        """
        prompt = SEARCH_MODE_PROMPT.format(question=question)

        try:
            response = self.llm.invoke(prompt)
            response = strip_think_tags(response).strip().upper()

            if "GLOBAL" in response:
                return "global"
            return "local"
        except Exception as e:
            logger.warning(f"Failed to determine search mode: {e}, defaulting to local")
            return "local"

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
            response = self.llm.invoke(prompt)
            response = strip_think_tags(response)

            if "NONE" in response.upper():
                return []

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

            # Partial match
            for node_id in self.kg.graph.nodes():
                if normalized in node_id or node_id in normalized:
                    matched_nodes.append(node_id)
                    break

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

        if community_context:
            context = f"=== COMMUNITY CONTEXT ==={community_context}\n\n{context}"

        return {
            "context": context,
            "mode": "local",
            "matched_entities": query_entities,
            "matched_nodes": list(matched_nodes),
            "total_nodes": len(all_relevant_nodes),
            "total_edges": subgraph.num_edges,
        }

    def global_search(self, question: str) -> dict:
        """
        Perform global search using community summaries.

        Args:
            question: User question

        Returns:
            Dictionary with retrieved context and metadata
        """
        if not self.community_vectordb:
            logger.warning(
                "Community vector store not initialized, falling back to local search"
            )
            return self.local_search(question)

        # Search community summaries
        try:
            results = self.community_vectordb.similarity_search_with_score(
                question, k=self.global_top_communities
            )
        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return self.local_search(question)

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

    def retrieve(
        self,
        question: str,
        mode: Literal["auto", "local", "global"] = "auto",
    ) -> dict:
        """
        Retrieve context for a question.

        Args:
            question: User question
            mode: Search mode ("auto", "local", or "global")

        Returns:
            Dictionary with context and metadata
        """
        # Determine mode if auto
        if mode == "auto":
            mode = self.determine_search_mode(question)
            logger.info(f"Auto-selected search mode: {mode}")

        # Perform search
        if mode == "global":
            result = self.global_search(question)
        else:
            result = self.local_search(question)

        logger.info(
            f"Retrieved context using {result['mode']} search: "
            f"{len(result['context'])} chars"
        )

        return result


def create_retriever(
    kg: KnowledgeGraph,
    communities: dict[str, Community],
    cfg,
) -> GraphRetriever:
    """Create a GraphRetriever from config."""
    vector_db_path = Path(cfg.PATHS.vector_db_dir) / "communities"

    return GraphRetriever(
        kg=kg,
        communities=communities,
        llm_model=cfg.EXTRACTION.model,
        embedding_model=cfg.EXTRACTION.embedding_model,
        vector_db_path=vector_db_path,
        local_search_hops=cfg.RETRIEVAL.local_search_hops,
        local_max_entities=cfg.RETRIEVAL.local_max_entities,
        local_max_edges=cfg.RETRIEVAL.get("local_max_edges", 50),
        global_top_communities=cfg.RETRIEVAL.global_top_communities,
    )
