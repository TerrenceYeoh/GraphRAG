"""
Knowledge Graph Builder Module

Constructs a NetworkX graph from extracted entities and relationships.
Provides persistence and loading functionality.
"""

from pathlib import Path
from typing import Iterator

import networkx as nx
from loguru import logger

from src.utils.graph_utils import (Entity, Relationship, load_json,
                                   normalize_entity_name, save_json)


class KnowledgeGraph:
    """
    Knowledge graph built from extracted entities and relationships.
    Uses NetworkX for in-memory graph operations with JSON persistence.
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.Graph()  # Undirected for community detection
        self.directed_graph = nx.DiGraph()  # Directed for relationship traversal
        self._entity_map: dict[str, Entity] = {}  # normalized_name -> Entity

    @property
    def num_nodes(self) -> int:
        """Return number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Return number of edges in the graph."""
        return self.graph.number_of_edges()

    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity as a node to the graph.

        Args:
            entity: Entity to add

        Returns:
            Normalized node ID
        """
        node_id = normalize_entity_name(entity.name)

        # Store entity mapping
        if node_id in self._entity_map:
            # Merge with existing entity
            existing = self._entity_map[node_id]
            if entity.description and entity.description not in existing.description:
                existing.description = (
                    f"{existing.description} {entity.description}".strip()
                )
            existing.attributes.update(entity.attributes)
        else:
            self._entity_map[node_id] = entity

        # Add/update node in both graphs
        node_attrs = {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "source_chunk": entity.source_chunk,
            **entity.attributes,
        }

        self.graph.add_node(node_id, **node_attrs)
        self.directed_graph.add_node(node_id, **node_attrs)

        return node_id

    def add_relationship(self, relationship: Relationship) -> tuple[str, str] | None:
        """
        Add a relationship as an edge to the graph.

        Args:
            relationship: Relationship to add

        Returns:
            Tuple of (source_id, target_id) or None if entities don't exist
        """
        source_id = normalize_entity_name(relationship.source)
        target_id = normalize_entity_name(relationship.target)

        # Check if both nodes exist
        if source_id not in self.graph or target_id not in self.graph:
            logger.warning(
                f"Skipping relationship: {relationship.source} -> {relationship.target} "
                f"(missing nodes: source={source_id in self.graph}, target={target_id in self.graph})"
            )
            return None

        edge_attrs = {
            "relation_type": relationship.relation_type,
            "description": relationship.description,
            "weight": relationship.weight,
            "source_chunk": relationship.source_chunk,
            **relationship.attributes,
        }

        # Add edge to undirected graph (combine weights if exists)
        if self.graph.has_edge(source_id, target_id):
            self.graph[source_id][target_id]["weight"] += relationship.weight
        else:
            self.graph.add_edge(source_id, target_id, **edge_attrs)

        # Add edge to directed graph
        if self.directed_graph.has_edge(source_id, target_id):
            self.directed_graph[source_id][target_id]["weight"] += relationship.weight
        else:
            self.directed_graph.add_edge(source_id, target_id, **edge_attrs)

        return source_id, target_id

    def add_entities(self, entities: list[Entity]) -> None:
        """Add multiple entities to the graph."""
        for entity in entities:
            self.add_entity(entity)
        logger.info(
            f"Added {len(entities)} entities, graph now has {self.num_nodes} nodes"
        )

    def add_relationships(self, relationships: list[Relationship]) -> None:
        """Add multiple relationships to the graph."""
        added = 0
        for rel in relationships:
            if self.add_relationship(rel):
                added += 1
        logger.info(
            f"Added {added}/{len(relationships)} relationships, graph now has {self.num_edges} edges"
        )

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name."""
        node_id = normalize_entity_name(name)
        return self._entity_map.get(node_id)

    def get_node_data(self, name: str) -> dict | None:
        """Get node data from graph."""
        node_id = normalize_entity_name(name)
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_neighbors(self, name: str, hops: int = 1) -> list[str]:
        """
        Get neighbors of a node within specified hops.

        Args:
            name: Entity name
            hops: Number of hops to traverse

        Returns:
            List of neighbor node IDs
        """
        node_id = normalize_entity_name(name)
        if node_id not in self.graph:
            return []

        neighbors = set()
        current_level = {node_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor != node_id and neighbor not in neighbors:
                        next_level.add(neighbor)
            neighbors.update(next_level)
            current_level = next_level

        return list(neighbors)

    def get_subgraph(self, node_names: list[str], hops: int = 1) -> "KnowledgeGraph":
        """
        Extract a subgraph around specified nodes.

        Args:
            node_names: List of entity names
            hops: Number of hops to include

        Returns:
            New KnowledgeGraph containing the subgraph
        """
        # Get all nodes within hops
        node_ids = set()
        for name in node_names:
            node_id = normalize_entity_name(name)
            if node_id in self.graph:
                node_ids.add(node_id)
                node_ids.update(self.get_neighbors(name, hops))

        # Create subgraph
        subgraph = KnowledgeGraph()
        subgraph.graph = self.graph.subgraph(node_ids).copy()
        subgraph.directed_graph = self.directed_graph.subgraph(node_ids).copy()

        # Copy entity map for relevant nodes
        for node_id in node_ids:
            if node_id in self._entity_map:
                subgraph._entity_map[node_id] = self._entity_map[node_id]

        return subgraph

    def get_edges_for_node(self, name: str) -> list[dict]:
        """Get all edges connected to a node."""
        node_id = normalize_entity_name(name)
        if node_id not in self.directed_graph:
            return []

        edges = []

        # Outgoing edges
        for _, target, data in self.directed_graph.out_edges(node_id, data=True):
            edges.append(
                {
                    "source": node_id,
                    "target": target,
                    "direction": "outgoing",
                    **data,
                }
            )

        # Incoming edges
        for source, _, data in self.directed_graph.in_edges(node_id, data=True):
            edges.append(
                {
                    "source": source,
                    "target": node_id,
                    "direction": "incoming",
                    **data,
                }
            )

        return edges

    def get_all_nodes(self) -> Iterator[tuple[str, dict]]:
        """Iterate over all nodes with their data."""
        for node_id, data in self.graph.nodes(data=True):
            yield node_id, data

    def get_all_edges(self) -> Iterator[tuple[str, str, dict]]:
        """Iterate over all edges with their data."""
        for source, target, data in self.directed_graph.edges(data=True):
            yield source, target, data

    def get_node_degrees(self) -> dict[str, int]:
        """Get degree (number of connections) for each node."""
        return dict(self.graph.degree())

    def get_connected_components(self) -> list[set[str]]:
        """Get list of connected components."""
        return [set(c) for c in nx.connected_components(self.graph)]

    def save(self, path: Path) -> None:
        """
        Save graph to JSON files.

        Args:
            path: Directory to save graph files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save nodes
        nodes_data = []
        for node_id, data in self.graph.nodes(data=True):
            nodes_data.append({"id": node_id, **data})
        save_json(nodes_data, path / "nodes.json")

        # Save edges
        edges_data = []
        for source, target, data in self.directed_graph.edges(data=True):
            edges_data.append({"source": source, "target": target, **data})
        save_json(edges_data, path / "edges.json")

        # Save entity map
        entity_map_data = {k: v.to_dict() for k, v in self._entity_map.items()}
        save_json(entity_map_data, path / "entity_map.json")

        logger.info(
            f"Saved graph to {path}: {self.num_nodes} nodes, {self.num_edges} edges"
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        """
        Load graph from JSON files.

        Args:
            path: Directory containing graph files

        Returns:
            Loaded KnowledgeGraph
        """
        path = Path(path)
        kg = cls()

        # Load nodes
        nodes_path = path / "nodes.json"
        if nodes_path.exists():
            nodes_data = load_json(nodes_path)
            for node in nodes_data:
                node_id = node.pop("id")
                kg.graph.add_node(node_id, **node)
                kg.directed_graph.add_node(node_id, **node)

        # Load edges
        edges_path = path / "edges.json"
        if edges_path.exists():
            edges_data = load_json(edges_path)
            for edge in edges_data:
                source = edge.pop("source")
                target = edge.pop("target")
                kg.graph.add_edge(source, target, **edge)
                kg.directed_graph.add_edge(source, target, **edge)

        # Load entity map
        entity_map_path = path / "entity_map.json"
        if entity_map_path.exists():
            entity_map_data = load_json(entity_map_path)
            kg._entity_map = {
                k: Entity.from_dict(v) for k, v in entity_map_data.items()
            }

        logger.info(
            f"Loaded graph from {path}: {kg.num_nodes} nodes, {kg.num_edges} edges"
        )
        return kg

    def to_context_string(self, max_nodes: int = 20, max_edges: int = 30) -> str:
        """
        Convert graph to a string representation for LLM context.

        Args:
            max_nodes: Maximum number of nodes to include
            max_edges: Maximum number of edges to include

        Returns:
            String representation of the graph
        """
        lines = []

        # Add entities
        lines.append("=== ENTITIES ===")
        node_degrees = self.get_node_degrees()
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

        for node_id, degree in sorted_nodes[:max_nodes]:
            data = self.graph.nodes[node_id]
            name = data.get("name", node_id)
            entity_type = data.get("type", "Unknown")
            description = data.get("description", "")
            lines.append(f"- {name} ({entity_type}): {description[:200]}")

        # Add relationships
        lines.append("\n=== RELATIONSHIPS ===")
        edges = list(self.directed_graph.edges(data=True))
        # Sort by weight
        edges.sort(key=lambda x: x[2].get("weight", 1), reverse=True)

        for source, target, data in edges[:max_edges]:
            rel_type = data.get("relation_type", "RELATED_TO")
            source_name = self.graph.nodes[source].get("name", source)
            target_name = self.graph.nodes[target].get("name", target)
            lines.append(f"- ({source_name}) --[{rel_type}]--> ({target_name})")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_connected_components": len(self.get_connected_components()),
            "avg_degree": sum(dict(self.graph.degree()).values())
            / max(self.num_nodes, 1),
            "density": nx.density(self.graph) if self.num_nodes > 0 else 0,
            "entity_types": self._count_entity_types(),
            "relationship_types": self._count_relationship_types(),
        }

    def _count_entity_types(self) -> dict[str, int]:
        """
        Count entities by type.

        Returns:
            Dictionary mapping entity type names to their counts
        """
        counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get("type", "Unknown")
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts

    def _count_relationship_types(self) -> dict[str, int]:
        """
        Count relationships by type.

        Returns:
            Dictionary mapping relationship type names to their counts
        """
        counts: dict[str, int] = {}
        for _, _, data in self.directed_graph.edges(data=True):
            rel_type = data.get("relation_type", "UNKNOWN")
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts


def build_graph_from_extractions(
    entities: list[Entity],
    relationships: list[Relationship],
) -> KnowledgeGraph:
    """
    Build a knowledge graph from extracted entities and relationships.

    Args:
        entities: List of extracted entities
        relationships: List of extracted relationships

    Returns:
        Constructed KnowledgeGraph
    """
    kg = KnowledgeGraph()
    kg.add_entities(entities)
    kg.add_relationships(relationships)

    # Log statistics
    stats = kg.get_statistics()
    logger.info(f"Built knowledge graph: {stats}")

    return kg
