"""
Entity and Relationship Extraction Module

Uses LLM to extract entities and relationships from text chunks.
This is the core component of Graph RAG that transforms unstructured
text into structured knowledge graph elements.
"""

import json
import re
from pathlib import Path

from langchain_ollama import OllamaLLM
from loguru import logger
from tqdm import tqdm

from src.utils.graph_utils import (Entity, ExtractionResult, Relationship,
                                   load_json, merge_entities,
                                   merge_relationships, save_json,
                                   strip_think_tags)

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from text.

Given the following text, extract all entities and relationships between them.

ENTITY TYPES to look for:
{entity_types}

For each ENTITY, provide:
- name: The entity's name (be specific, use full names)
- type: One of the entity types listed above
- description: A brief description of the entity based on the context

For each RELATIONSHIP, provide:
- source: The source entity name (must match an extracted entity)
- target: The target entity name (must match an extracted entity)
- relation_type: The type of relationship (use UPPERCASE_WITH_UNDERSCORES, e.g., WORKS_FOR, PART_OF, REQUIRES)
- description: A brief description of the relationship

TEXT:
{text}

Respond with ONLY valid JSON in this exact format (no other text):
{{
    "entities": [
        {{"name": "...", "type": "...", "description": "..."}},
        ...
    ],
    "relationships": [
        {{"source": "...", "target": "...", "relation_type": "...", "description": "..."}},
        ...
    ]
}}

IMPORTANT:
- Extract ALL relevant entities and relationships
- Entity names should be consistent (same entity = same name)
- Only include relationships where both source and target are in your entities list
- If no entities or relationships found, return empty lists
- Respond with ONLY the JSON, no explanations
"""


def repair_json(json_str: str) -> str:
    """Attempt to repair common JSON formatting issues from LLM output."""
    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Fix unquoted keys (common LLM error)
    json_str = re.sub(r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1 "\2":', json_str)

    # Replace single quotes with double quotes (but not inside strings)
    # This is a simplified approach - handles most cases
    json_str = re.sub(r"(?<![a-zA-Z])'([^']*)'(?![a-zA-Z])", r'"\1"', json_str)

    # Remove control characters except newlines and tabs
    json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_str)

    # Fix common escaping issues
    json_str = json_str.replace("\\'", "'")

    # Handle truncated JSON - try to close unclosed brackets
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Try to find a valid JSON ending point
        # Look for the last complete entity or relationship
        last_complete = max(
            json_str.rfind("}},"),
            json_str.rfind("}]"),
            json_str.rfind('"}'),
        )
        if last_complete > 0:
            json_str = json_str[: last_complete + 2]
            # Close remaining brackets
            json_str += "]}" * open_braces

    return json_str


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling various formats and errors."""
    # First strip think tags
    response = strip_think_tags(response)

    # Strategy 1: Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        candidate = json_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # Try other strategies

    # Strategy 2: Find the outermost JSON object
    # Match from first { to last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = response[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Strategy 3: Try repairing the JSON
            try:
                repaired = repair_json(candidate)
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Try to extract entities and relationships arrays separately
    entities = []
    relationships = []

    # Find entities array
    entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', response, re.DOTALL)
    if entities_match:
        try:
            entities_str = "[" + entities_match.group(1) + "]"
            entities_str = repair_json(entities_str)
            entities = json.loads(entities_str)
        except json.JSONDecodeError:
            pass

    # Find relationships array
    rels_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', response, re.DOTALL)
    if rels_match:
        try:
            rels_str = "[" + rels_match.group(1) + "]"
            rels_str = repair_json(rels_str)
            relationships = json.loads(rels_str)
        except json.JSONDecodeError:
            pass

    if entities or relationships:
        return {"entities": entities, "relationships": relationships}

    # All strategies failed
    logger.warning(f"Failed to parse JSON from response (length: {len(response)})")
    logger.debug(f"Response was: {response[:500]}...")
    return {"entities": [], "relationships": []}


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        entity_types: list[str] | None = None,
        temperature: float = 0.0,
        cache_dir: Path | None = None,
    ):
        """
        Initialize the entity extractor.

        Args:
            model: Ollama model name for extraction
            entity_types: List of entity types to extract
            temperature: LLM temperature (lower = more deterministic)
            cache_dir: Directory to cache extraction results
        """
        self.model = model
        self.entity_types = entity_types or [
            "Person",
            "Organization",
            "Program",
            "Skill",
            "Technology",
            "Date",
            "Duration",
            "Location",
            "Concept",
            "Requirement",
        ]
        self.temperature = temperature
        self.cache_dir = cache_dir

        self.llm = OllamaLLM(
            model=model,
            temperature=temperature,
        )

        logger.info(f"Initialized EntityExtractor with model: {model}")

    def _get_cache_path(self, chunk_id: str) -> Path | None:
        """Get cache file path for a chunk."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{chunk_id}.json"

    def _load_from_cache(self, chunk_id: str) -> ExtractionResult | None:
        """Load extraction result from cache if available."""
        cache_path = self._get_cache_path(chunk_id)
        if cache_path and cache_path.exists():
            try:
                data = load_json(cache_path)
                return ExtractionResult.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {chunk_id}: {e}")
        return None

    def _save_to_cache(self, chunk_id: str, result: ExtractionResult) -> None:
        """Save extraction result to cache."""
        cache_path = self._get_cache_path(chunk_id)
        if cache_path:
            save_json(result.to_dict(), cache_path)

    def extract_from_chunk(
        self,
        text: str,
        chunk_id: str = "",
        use_cache: bool = True,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a text chunk.

        Args:
            text: The text chunk to process
            chunk_id: Unique identifier for the chunk (for caching)
            use_cache: Whether to use cached results if available

        Returns:
            ExtractionResult containing entities and relationships
        """
        # Check cache first
        if use_cache and chunk_id:
            cached = self._load_from_cache(chunk_id)
            if cached:
                logger.debug(f"Loaded from cache: {chunk_id}")
                return cached

        # Format the prompt
        entity_types_str = "\n".join(f"- {et}" for et in self.entity_types)
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=entity_types_str,
            text=text,
        )

        # Call LLM
        try:
            response = self.llm.invoke(prompt)
            parsed = extract_json_from_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed for chunk {chunk_id}: {e}")
            parsed = {"entities": [], "relationships": []}

        # Parse entities
        entities = []
        for ent_data in parsed.get("entities", []):
            try:
                entity = Entity(
                    name=ent_data.get("name", "").strip(),
                    type=ent_data.get("type", "Concept").strip(),
                    description=ent_data.get("description", "").strip(),
                    source_chunk=chunk_id,
                )
                if entity.name:  # Only add if name is not empty
                    entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity: {ent_data}, error: {e}")

        # Parse relationships
        relationships = []
        entity_names = {e.name.lower() for e in entities}
        for rel_data in parsed.get("relationships", []):
            try:
                source = rel_data.get("source", "").strip()
                target = rel_data.get("target", "").strip()

                # Only include relationships where both entities exist
                if source.lower() in entity_names and target.lower() in entity_names:
                    relationship = Relationship(
                        source=source,
                        target=target,
                        relation_type=rel_data.get("relation_type", "RELATED_TO")
                        .strip()
                        .upper(),
                        description=rel_data.get("description", "").strip(),
                        source_chunk=chunk_id,
                    )
                    relationships.append(relationship)
            except Exception as e:
                logger.warning(f"Failed to parse relationship: {rel_data}, error: {e}")

        result = ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_text=text[:500] + "..." if len(text) > 500 else text,
            chunk_id=chunk_id,
        )

        # Cache result
        if chunk_id:
            self._save_to_cache(chunk_id, result)

        logger.debug(
            f"Extracted {len(entities)} entities, {len(relationships)} relationships from chunk {chunk_id}"
        )
        return result

    def extract_from_chunks(
        self,
        chunks: list[tuple[str, str]],  # (chunk_id, text)
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> list[ExtractionResult]:
        """
        Extract entities and relationships from multiple chunks.

        Args:
            chunks: List of (chunk_id, text) tuples
            use_cache: Whether to use cached results
            show_progress: Whether to show progress bar

        Returns:
            List of ExtractionResult objects
        """
        results = []
        iterator = tqdm(chunks, desc="Extracting entities") if show_progress else chunks

        for chunk_id, text in iterator:
            result = self.extract_from_chunk(text, chunk_id, use_cache)
            results.append(result)

        # Log summary
        total_entities = sum(len(r.entities) for r in results)
        total_relationships = sum(len(r.relationships) for r in results)
        logger.info(
            f"Extraction complete: {total_entities} entities, {total_relationships} relationships from {len(chunks)} chunks"
        )

        return results

    def merge_results(
        self, results: list[ExtractionResult]
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Merge extraction results, deduplicating entities and relationships.

        Args:
            results: List of ExtractionResult objects

        Returns:
            Tuple of (merged_entities, merged_relationships)
        """
        all_entities = []
        all_relationships = []

        for result in results:
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)

        merged_entities = merge_entities(all_entities)
        merged_relationships = merge_relationships(all_relationships)

        logger.info(
            f"Merged to {len(merged_entities)} unique entities, {len(merged_relationships)} unique relationships"
        )

        return merged_entities, merged_relationships


def create_extractor(cfg) -> EntityExtractor:
    """Create an EntityExtractor from config."""
    cache_dir = Path(cfg.PATHS.graph_db_dir) / "extraction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return EntityExtractor(
        model=cfg.EXTRACTION.model,
        entity_types=list(cfg.EXTRACTION.entity_types),
        temperature=cfg.EXTRACTION.temperature,
        cache_dir=cache_dir,
    )
