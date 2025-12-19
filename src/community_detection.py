"""
Community Detection Module

Uses the Leiden algorithm to detect hierarchical communities in the knowledge graph.
Communities are groups of densely connected entities that can be summarized together.
"""

from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger

try:
    from graspologic.partition import hierarchical_leiden
except ImportError:
    hierarchical_leiden = None
    logger.warning(
        "graspologic not installed, falling back to NetworkX community detection"
    )

from src.graph_builder import KnowledgeGraph
from src.utils.graph_utils import Community, load_json, save_json


class CommunityDetector:
    """
    Detects hierarchical communities in a knowledge graph using the Leiden algorithm.
    """

    def __init__(
        self,
        max_levels: int = 2,
        min_community_size: int = 3,
        resolution: float = 1.0,
        random_seed: int = 42,
    ):
        """
        Initialize community detector.

        Args:
            max_levels: Maximum levels of hierarchy to detect
            min_community_size: Minimum nodes in a community
            resolution: Resolution parameter for Leiden (higher = more communities)
            random_seed: Random seed for reproducibility
        """
        self.max_levels = max_levels
        self.min_community_size = min_community_size
        self.resolution = resolution
        self.random_seed = random_seed

    def detect_communities(self, kg: KnowledgeGraph) -> dict[str, Community]:
        """
        Detect communities in the knowledge graph.

        Args:
            kg: KnowledgeGraph to analyze

        Returns:
            Dictionary mapping community_id to Community object
        """
        if kg.num_nodes < self.min_community_size:
            logger.warning(
                f"Graph too small ({kg.num_nodes} nodes) for community detection"
            )
            return self._create_single_community(kg)

        if hierarchical_leiden is not None:
            return self._detect_with_leiden(kg)
        else:
            return self._detect_with_louvain(kg)

    def _detect_with_leiden(self, kg: KnowledgeGraph) -> dict[str, Community]:
        """Detect communities using hierarchical Leiden algorithm."""
        logger.info("Running hierarchical Leiden community detection...")

        # Convert to numpy adjacency matrix for graspologic
        nodes = list(kg.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        n = len(nodes)

        if n == 0:
            return {}

        # Build adjacency matrix
        adj_matrix = np.zeros((n, n))
        for source, target, data in kg.graph.edges(data=True):
            i, j = node_to_idx[source], node_to_idx[target]
            weight = data.get("weight", 1.0)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Undirected

        # Run hierarchical Leiden
        try:
            hierarchical_clusters = hierarchical_leiden(
                adj_matrix,
                max_cluster_size=max(n // 2, self.min_community_size * 2),
                random_seed=self.random_seed,
            )
        except Exception as e:
            logger.error(f"Leiden algorithm failed: {e}, falling back to Louvain")
            return self._detect_with_louvain(kg)

        # Parse HierarchicalClusters results into Community objects
        communities: dict[str, Community] = {}

        # Get the final clustering (level 0 in our hierarchy)
        try:
            final_clusters = hierarchical_clusters.final_level_hierarchical_clustering()
            # final_clusters is a dict: {node_idx: cluster_id}
            # Only contains non-isolate nodes

            # Group nodes by cluster
            cluster_nodes: dict[int, list[str]] = {}
            for node_idx, cluster_id in final_clusters.items():
                node_id = idx_to_node.get(node_idx)
                if node_id:
                    if cluster_id not in cluster_nodes:
                        cluster_nodes[cluster_id] = []
                    cluster_nodes[cluster_id].append(node_id)

            # Create communities
            for cluster_id, node_ids in cluster_nodes.items():
                if len(node_ids) >= self.min_community_size:
                    community_id = f"community_0_{cluster_id}"
                    community = Community(
                        id=community_id,
                        level=0,
                        entity_ids=node_ids,
                        title=self._generate_community_title(set(node_ids), kg),
                    )
                    communities[community_id] = community

            logger.info(f"Detected {len(communities)} communities using Leiden")

        except Exception as e:
            logger.error(
                f"Failed to parse Leiden results: {e}, falling back to Louvain"
            )
            return self._detect_with_louvain(kg)

        return communities

    def _detect_with_louvain(self, kg: KnowledgeGraph) -> dict[str, Community]:
        """Detect communities using Louvain algorithm (fallback)."""
        logger.info("Running Louvain community detection...")

        communities: dict[str, Community] = {}

        # Level 0: Louvain communities
        try:
            partition = nx.community.louvain_communities(
                kg.graph,
                resolution=self.resolution,
                seed=self.random_seed,
            )
        except Exception as e:
            logger.error(f"Louvain failed: {e}")
            return self._create_single_community(kg)

        level_0_communities = []
        for i, node_set in enumerate(partition):
            if len(node_set) >= self.min_community_size:
                community_id = f"community_0_{i}"
                community = Community(
                    id=community_id,
                    level=0,
                    entity_ids=list(node_set),
                    title=self._generate_community_title(node_set, kg),
                )
                communities[community_id] = community
                level_0_communities.append(community)

        # Level 1: Merge small communities into larger ones
        if self.max_levels > 1 and len(level_0_communities) > 1:
            level_1_communities = self._create_higher_level_communities(
                level_0_communities, kg, level=1
            )
            for comm in level_1_communities:
                communities[comm.id] = comm

        logger.info(f"Detected {len(communities)} communities using Louvain")
        return communities

    def _parse_leiden_results(
        self,
        community_mapping: dict,
        nodes: list[str],
        kg: KnowledgeGraph,
    ) -> dict[str, Community]:
        """Parse hierarchical Leiden results into Community objects."""
        communities: dict[str, Community] = {}

        # Group nodes by their community assignments at each level
        # community_mapping is {node_idx: [level0_comm, level1_comm, ...]}
        level_communities: dict[int, dict[int, list[str]]] = {}

        for node_idx, comm_path in community_mapping.items():
            node_id = nodes[node_idx]
            for level, comm_id in enumerate(comm_path):
                if level > self.max_levels:
                    break
                if level not in level_communities:
                    level_communities[level] = {}
                if comm_id not in level_communities[level]:
                    level_communities[level][comm_id] = []
                level_communities[level][comm_id].append(node_id)

        # Create Community objects
        parent_map: dict[tuple[int, int], str] = {}  # (level, comm_id) -> community_id

        for level in sorted(level_communities.keys()):
            for comm_idx, node_ids in level_communities[level].items():
                if len(node_ids) < self.min_community_size:
                    continue

                community_id = f"community_{level}_{comm_idx}"
                parent_id = None

                # Find parent community
                if level > 0 and (level - 1, comm_idx) in parent_map:
                    parent_id = parent_map[(level - 1, comm_idx)]

                community = Community(
                    id=community_id,
                    level=level,
                    entity_ids=node_ids,
                    title=self._generate_community_title(set(node_ids), kg),
                    parent_id=parent_id,
                )
                communities[community_id] = community
                parent_map[(level, comm_idx)] = community_id

                # Update parent's child list
                if parent_id and parent_id in communities:
                    communities[parent_id].child_ids.append(community_id)

        return communities

    def _create_higher_level_communities(
        self,
        lower_communities: list[Community],
        kg: KnowledgeGraph,
        level: int,
    ) -> list[Community]:
        """Create higher-level communities by grouping lower-level ones."""
        if len(lower_communities) <= 1:
            return []

        # Build a graph of communities based on inter-community edges
        comm_graph = nx.Graph()
        for comm in lower_communities:
            comm_graph.add_node(comm.id)

        # Build node-to-community mapping for O(1) lookups
        node_to_comm: dict[str, str] = {}
        for comm in lower_communities:
            for node_id in comm.entity_ids:
                node_to_comm[node_id] = comm.id

        # Count inter-community edges in O(E) by iterating edges once
        edge_counts: dict[tuple[str, str], int] = {}
        for source, target in kg.graph.edges():
            comm1_id = node_to_comm.get(source)
            comm2_id = node_to_comm.get(target)

            # Only count edges between different communities
            if comm1_id and comm2_id and comm1_id != comm2_id:
                # Normalize edge key (smaller id first) to avoid duplicates
                edge_key = (min(comm1_id, comm2_id), max(comm1_id, comm2_id))
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        # Add edges to community graph
        for (comm1_id, comm2_id), weight in edge_counts.items():
            comm_graph.add_edge(comm1_id, comm2_id, weight=weight)

        # Run Louvain on community graph
        try:
            partition = nx.community.louvain_communities(
                comm_graph,
                resolution=self.resolution / 2,  # Lower resolution for higher level
                seed=self.random_seed,
            )
        except Exception:
            return []

        higher_communities = []
        for i, comm_set in enumerate(partition):
            if len(comm_set) < 2:
                continue

            # Merge entities from all communities in this group
            merged_entities = []
            child_ids = []
            for comm_id in comm_set:
                for lower_comm in lower_communities:
                    if lower_comm.id == comm_id:
                        merged_entities.extend(lower_comm.entity_ids)
                        child_ids.append(comm_id)
                        lower_comm.parent_id = f"community_{level}_{i}"

            if len(merged_entities) >= self.min_community_size:
                community = Community(
                    id=f"community_{level}_{i}",
                    level=level,
                    entity_ids=list(set(merged_entities)),
                    title=self._generate_community_title(set(merged_entities), kg),
                    child_ids=child_ids,
                )
                higher_communities.append(community)

        return higher_communities

    def _create_single_community(self, kg: KnowledgeGraph) -> dict[str, Community]:
        """Create a single community containing all nodes."""
        if kg.num_nodes == 0:
            return {}

        node_ids = list(kg.graph.nodes())
        community = Community(
            id="community_0_0",
            level=0,
            entity_ids=node_ids,
            title=self._generate_community_title(set(node_ids), kg),
        )
        return {community.id: community}

    def _generate_community_title(self, node_ids: set[str], kg: KnowledgeGraph) -> str:
        """Generate a title for a community based on its entities."""
        # Get entity types and most connected nodes
        type_counts: dict[str, int] = {}
        node_degrees = []

        for node_id in node_ids:
            if node_id in kg.graph:
                data = kg.graph.nodes[node_id]
                entity_type = data.get("type", "Unknown")
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                node_degrees.append((node_id, kg.graph.degree(node_id)))

        # Get dominant type
        dominant_type = (
            max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Mixed"
        )

        # Get top 3 nodes by degree
        top_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)[:3]
        top_names = [kg.graph.nodes[n[0]].get("name", n[0]) for n in top_nodes]

        return f"{dominant_type} cluster: {', '.join(top_names)}"


def save_communities(communities: dict[str, Community], path: Path) -> None:
    """Save communities to JSON file."""
    data = {k: v.to_dict() for k, v in communities.items()}
    save_json(data, path)
    logger.info(f"Saved {len(communities)} communities to {path}")


def load_communities(path: Path) -> dict[str, Community]:
    """Load communities from JSON file."""
    data = load_json(path)
    communities = {k: Community.from_dict(v) for k, v in data.items()}
    logger.info(f"Loaded {len(communities)} communities from {path}")
    return communities


def create_detector(cfg) -> CommunityDetector:
    """Create a CommunityDetector from config."""
    return CommunityDetector(
        max_levels=cfg.COMMUNITY.max_levels,
        min_community_size=cfg.COMMUNITY.min_community_size,
        resolution=cfg.COMMUNITY.resolution,
        random_seed=cfg.COMMUNITY.random_seed,
    )
