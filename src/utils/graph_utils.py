"""Graph utility functions for Graph RAG"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> tags from LLM output.

    Many LLMs (especially reasoning models) wrap their chain-of-thought
    in <think> tags. This function removes them to get the final answer.

    Args:
        text: LLM output text potentially containing think tags

    Returns:
        Text with think tags and their content removed
    """
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned.strip()


@dataclass
class Entity:
    """Represents an entity node in the knowledge graph."""

    name: str
    type: str
    description: str = ""
    source_chunk: str = ""
    attributes: dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.name.lower(), self.type.lower()))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (
            self.name.lower() == other.name.lower()
            and self.type.lower() == other.type.lower()
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class Relationship:
    """Represents a relationship edge in the knowledge graph."""

    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    source_chunk: str = ""
    attributes: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(
            (self.source.lower(), self.target.lower(), self.relation_type.lower())
        )

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return (
            self.source.lower() == other.source.lower()
            and self.target.lower() == other.target.lower()
            and self.relation_type.lower() == other.relation_type.lower()
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        return cls(**data)


@dataclass
class ExtractionResult:
    """Result of entity and relationship extraction from a chunk."""

    entities: list[Entity]
    relationships: list[Relationship]
    source_text: str = ""
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "source_text": self.source_text,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            relationships=[
                Relationship.from_dict(r) for r in data.get("relationships", [])
            ],
            source_text=data.get("source_text", ""),
            chunk_id=data.get("chunk_id", ""),
        )


@dataclass
class Community:
    """Represents a community of related entities."""

    id: str
    level: int
    entity_ids: list[str]
    title: str = ""
    summary: str = ""
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Community":
        return cls(**data)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"Saved JSON to {path}")


def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for consistent matching."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """Merge duplicate entities, combining their descriptions."""
    entity_map: dict[tuple[str, str], Entity] = {}

    for entity in entities:
        key = (normalize_entity_name(entity.name), entity.type.lower())

        if key in entity_map:
            existing = entity_map[key]
            # Merge descriptions
            if entity.description and entity.description not in existing.description:
                existing.description = (
                    f"{existing.description} {entity.description}".strip()
                )
            # Merge attributes
            existing.attributes.update(entity.attributes)
        else:
            entity_map[key] = Entity(
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_chunk=entity.source_chunk,
                attributes=entity.attributes.copy(),
            )

    return list(entity_map.values())


def merge_relationships(relationships: list[Relationship]) -> list[Relationship]:
    """Merge duplicate relationships, combining weights."""
    rel_map: dict[tuple[str, str, str], Relationship] = {}

    for rel in relationships:
        key = (
            normalize_entity_name(rel.source),
            normalize_entity_name(rel.target),
            rel.relation_type.lower(),
        )

        if key in rel_map:
            existing = rel_map[key]
            existing.weight += rel.weight
            if rel.description and rel.description not in existing.description:
                existing.description = (
                    f"{existing.description} {rel.description}".strip()
                )
        else:
            rel_map[key] = Relationship(
                source=rel.source,
                target=rel.target,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight,
                source_chunk=rel.source_chunk,
                attributes=rel.attributes.copy(),
            )

    return list(rel_map.values())
