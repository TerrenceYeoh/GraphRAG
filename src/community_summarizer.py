"""
Community Summarization Module

Uses LLM to generate summaries for each detected community.
These summaries enable global search by providing high-level overviews
of related entity clusters.
"""

from pathlib import Path

import ollama
from loguru import logger
from tqdm import tqdm

from src.graph_builder import KnowledgeGraph
from src.utils.graph_utils import (Community, load_json, save_json,
                                   strip_think_tags)

COMMUNITY_SUMMARY_PROMPT = """/no_think
You are an expert at summarizing information about groups of related entities.

Given the following entities and their relationships, write a comprehensive summary that:
1. Identifies the main theme or topic of this group
2. Describes the key entities and their roles
3. Explains the relationships between entities
4. Highlights any important patterns or insights

COMMUNITY INFORMATION:
{community_info}

Write a clear, informative summary in 2-3 paragraphs. Focus on the most important information.

SUMMARY:"""


class CommunitySummarizer:
    """
    Generates summaries for communities using LLM.
    Supports caching to enable resumable summarization.
    """

    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        max_tokens: int = 500,
        temperature: float = 0.3,
        include_entity_descriptions: bool = True,
        include_relationships: bool = True,
        cache_dir: Path | None = None,
    ):
        """
        Initialize community summarizer.

        Args:
            model: Ollama model name
            max_tokens: Maximum tokens in summary
            temperature: LLM temperature
            include_entity_descriptions: Whether to include entity descriptions in context
            include_relationships: Whether to include relationships in context
            cache_dir: Directory for caching summaries (enables resume on interrupt)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_entity_descriptions = include_entity_descriptions
        self.include_relationships = include_relationships
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CommunitySummarizer with model: {model}")
        if self.cache_dir:
            logger.info(f"Summary caching enabled: {self.cache_dir}")

    def _build_community_context(
        self,
        community: Community,
        kg: KnowledgeGraph,
        max_entities: int = 30,
        max_relationships: int = 50,
    ) -> str:
        """Build context string for a community."""
        lines = []

        # Community metadata
        lines.append(f"Community: {community.title}")
        lines.append(f"Level: {community.level}")
        lines.append(f"Number of entities: {len(community.entity_ids)}")
        lines.append("")

        # Entities
        lines.append("ENTITIES:")
        entity_data = []
        for entity_id in community.entity_ids:
            if entity_id in kg.graph:
                node_data = kg.graph.nodes[entity_id]
                entity_data.append(
                    {
                        "id": entity_id,
                        "name": node_data.get("name", entity_id),
                        "type": node_data.get("type", "Unknown"),
                        "description": node_data.get("description", ""),
                        "degree": kg.graph.degree(entity_id),
                    }
                )

        # Sort by degree (most connected first)
        entity_data.sort(key=lambda x: x["degree"], reverse=True)

        for entity in entity_data[:max_entities]:
            if self.include_entity_descriptions and entity["description"]:
                lines.append(
                    f"- {entity['name']} ({entity['type']}): {entity['description'][:200]}"
                )
            else:
                lines.append(f"- {entity['name']} ({entity['type']})")

        if len(entity_data) > max_entities:
            lines.append(f"  ... and {len(entity_data) - max_entities} more entities")

        # Relationships
        if self.include_relationships:
            lines.append("")
            lines.append("RELATIONSHIPS:")

            community_nodes = set(community.entity_ids)
            relationships = []

            for source, target, data in kg.directed_graph.edges(data=True):
                if source in community_nodes and target in community_nodes:
                    source_name = kg.graph.nodes[source].get("name", source)
                    target_name = kg.graph.nodes[target].get("name", target)
                    rel_type = data.get("relation_type", "RELATED_TO")
                    weight = data.get("weight", 1)
                    relationships.append(
                        {
                            "source": source_name,
                            "target": target_name,
                            "type": rel_type,
                            "weight": weight,
                        }
                    )

            # Sort by weight
            relationships.sort(key=lambda x: x["weight"], reverse=True)

            for rel in relationships[:max_relationships]:
                lines.append(
                    f"- ({rel['source']}) --[{rel['type']}]--> ({rel['target']})"
                )

            if len(relationships) > max_relationships:
                lines.append(
                    f"  ... and {len(relationships) - max_relationships} more relationships"
                )

        return "\n".join(lines)

    def _get_cache_path(self, community_id: str) -> Path | None:
        """Get cache file path for a community."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{community_id}.json"

    def _load_from_cache(self, community_id: str) -> str | None:
        """Load summary from cache if available."""
        cache_path = self._get_cache_path(community_id)
        if cache_path and cache_path.exists():
            try:
                data = load_json(cache_path)
                return data.get("summary")
            except Exception as e:
                logger.warning(f"Failed to load cache for {community_id}: {e}")
        return None

    def _save_to_cache(self, community_id: str, summary: str) -> None:
        """Save summary to cache."""
        cache_path = self._get_cache_path(community_id)
        if cache_path:
            try:
                save_json({"community_id": community_id, "summary": summary}, cache_path)
            except Exception as e:
                logger.warning(f"Failed to save cache for {community_id}: {e}")

    def summarize_community(
        self,
        community: Community,
        kg: KnowledgeGraph,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a summary for a community.

        Args:
            community: Community to summarize
            kg: Knowledge graph
            use_cache: Whether to use cached summaries if available

        Returns:
            Generated summary text
        """
        # Check cache first
        if use_cache:
            cached_summary = self._load_from_cache(community.id)
            if cached_summary:
                logger.debug(f"Loaded summary from cache for {community.id}")
                return cached_summary

        # Build context
        context = self._build_community_context(community, kg)

        # Generate summary
        prompt = COMMUNITY_SUMMARY_PROMPT.format(community_info=context)

        try:
            result = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
                think=False,  # Disable thinking mode for speed
            )
            response = result["response"]
            summary = strip_think_tags(response)

            # Clean up the summary
            summary = summary.strip()
            if summary.startswith("SUMMARY:"):
                summary = summary[8:].strip()

            logger.debug(f"Generated summary for {community.id}: {len(summary)} chars")

            # Save to cache
            if use_cache:
                self._save_to_cache(community.id, summary)

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {community.id}: {e}")
            return f"Community containing: {', '.join(community.entity_ids[:5])}"

    def summarize_all_communities(
        self,
        communities: dict[str, Community],
        kg: KnowledgeGraph,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Community]:
        """
        Generate summaries for all communities.

        Args:
            communities: Dictionary of communities
            kg: Knowledge graph
            show_progress: Whether to show progress bar
            use_cache: Whether to use cached summaries if available

        Returns:
            Communities with summaries added
        """
        # Sort by level (lower levels first, as higher levels may depend on them)
        sorted_communities = sorted(
            communities.values(),
            key=lambda c: (c.level, len(c.entity_ids)),
            reverse=True,  # Start with larger communities at each level
        )

        # Count cached summaries for progress reporting
        cached_count = 0
        generated_count = 0

        iterator = (
            tqdm(sorted_communities, desc="Summarizing communities")
            if show_progress
            else sorted_communities
        )

        for community in iterator:
            # Check if we'll use cache (for stats)
            was_cached = (
                use_cache
                and self.cache_dir
                and self._load_from_cache(community.id) is not None
            )

            summary = self.summarize_community(community, kg, use_cache=use_cache)
            community.summary = summary

            if was_cached:
                cached_count += 1
            else:
                generated_count += 1

        logger.info(
            f"Summarized {len(communities)} communities "
            f"({cached_count} from cache, {generated_count} generated)"
        )
        return communities


def save_community_summaries(communities: dict[str, Community], path: Path) -> None:
    """Save communities with summaries to JSON file."""
    data = {k: v.to_dict() for k, v in communities.items()}
    save_json(data, path)
    logger.info(f"Saved {len(communities)} community summaries to {path}")


def load_community_summaries(path: Path) -> dict[str, Community]:
    """Load communities with summaries from JSON file."""
    data = load_json(path)
    communities = {k: Community.from_dict(v) for k, v in data.items()}
    logger.info(f"Loaded {len(communities)} community summaries from {path}")
    return communities


def create_summarizer(cfg) -> CommunitySummarizer:
    """Create a CommunitySummarizer from config."""
    # Set up cache directory
    cache_dir = Path(cfg.PATHS.graph_db_dir) / "summary_cache"

    return CommunitySummarizer(
        model=cfg.SUMMARIZATION.model,
        max_tokens=cfg.SUMMARIZATION.max_tokens,
        temperature=cfg.SUMMARIZATION.temperature,
        include_entity_descriptions=cfg.SUMMARIZATION.include_entity_descriptions,
        include_relationships=cfg.SUMMARIZATION.include_relationships,
        cache_dir=cache_dir,
    )
