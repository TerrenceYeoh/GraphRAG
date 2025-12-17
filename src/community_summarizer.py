"""
Community Summarization Module

Uses LLM to generate summaries for each detected community.
These summaries enable global search by providing high-level overviews
of related entity clusters.
"""

from pathlib import Path

from langchain_ollama import OllamaLLM
from loguru import logger
from tqdm import tqdm

from src.utils.graph_utils import Community, save_json, load_json, strip_think_tags
from src.graph_builder import KnowledgeGraph


COMMUNITY_SUMMARY_PROMPT = """You are an expert at summarizing information about groups of related entities.

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
    """

    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        max_tokens: int = 500,
        temperature: float = 0.3,
        include_entity_descriptions: bool = True,
        include_relationships: bool = True,
    ):
        """
        Initialize community summarizer.

        Args:
            model: Ollama model name
            max_tokens: Maximum tokens in summary
            temperature: LLM temperature
            include_entity_descriptions: Whether to include entity descriptions in context
            include_relationships: Whether to include relationships in context
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_entity_descriptions = include_entity_descriptions
        self.include_relationships = include_relationships

        self.llm = OllamaLLM(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )

        logger.info(f"Initialized CommunitySummarizer with model: {model}")

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
                lines.append(f"- ({rel['source']}) --[{rel['type']}]--> ({rel['target']})")

            if len(relationships) > max_relationships:
                lines.append(
                    f"  ... and {len(relationships) - max_relationships} more relationships"
                )

        return "\n".join(lines)

    def summarize_community(
        self,
        community: Community,
        kg: KnowledgeGraph,
    ) -> str:
        """
        Generate a summary for a community.

        Args:
            community: Community to summarize
            kg: Knowledge graph

        Returns:
            Generated summary text
        """
        # Build context
        context = self._build_community_context(community, kg)

        # Generate summary
        prompt = COMMUNITY_SUMMARY_PROMPT.format(community_info=context)

        try:
            response = self.llm.invoke(prompt)
            summary = strip_think_tags(response)

            # Clean up the summary
            summary = summary.strip()
            if summary.startswith("SUMMARY:"):
                summary = summary[8:].strip()

            logger.debug(f"Generated summary for {community.id}: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {community.id}: {e}")
            return f"Community containing: {', '.join(community.entity_ids[:5])}"

    def summarize_all_communities(
        self,
        communities: dict[str, Community],
        kg: KnowledgeGraph,
        show_progress: bool = True,
    ) -> dict[str, Community]:
        """
        Generate summaries for all communities.

        Args:
            communities: Dictionary of communities
            kg: Knowledge graph
            show_progress: Whether to show progress bar

        Returns:
            Communities with summaries added
        """
        # Sort by level (lower levels first, as higher levels may depend on them)
        sorted_communities = sorted(
            communities.values(),
            key=lambda c: (c.level, len(c.entity_ids)),
            reverse=True,  # Start with larger communities at each level
        )

        iterator = (
            tqdm(sorted_communities, desc="Summarizing communities")
            if show_progress
            else sorted_communities
        )

        for community in iterator:
            summary = self.summarize_community(community, kg)
            community.summary = summary

        logger.info(f"Generated summaries for {len(communities)} communities")
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
    return CommunitySummarizer(
        model=cfg.SUMMARIZATION.model,
        max_tokens=cfg.SUMMARIZATION.max_tokens,
        temperature=cfg.SUMMARIZATION.temperature,
        include_entity_descriptions=cfg.SUMMARIZATION.include_entity_descriptions,
        include_relationships=cfg.SUMMARIZATION.include_relationships,
    )
