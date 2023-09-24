"""
This module contains our custom embedding model

The name is a placeholder
"""
import random
from typing import Union

import networkx as nx
import numpy as np

from local_simhash import CustomSimhash


class Livesketch:
    """
    A simhash based edge embedding model
    """

    def __init__(
        self,
        dimensions: int = 32,
        seed: int = 42,
        random_walk_length: Union[int, None] = None,
        number_of_walks: int = 1000,
        use_page_rank: bool = False,
    ):
        self.dimensions = dimensions
        self._sketch = None
        self.seed = seed
        self.hash_signatures = {}
        self.neighborhoods = {}

        if random_walk_length and use_page_rank:
            raise ValueError("You can only use one of random walk length and page rank")

        self.random_walk_length = random_walk_length
        self.number_of_walks = number_of_walks
        self.use_page_rank = use_page_rank

    def _set_seed(self):
        """
        Creating the initial random seed.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """
        Checking the consecutive numeric indexing
        """
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    @staticmethod
    def _ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """
        Ensure walk traversal conditions
        """
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """
        Check the Karate Club assumptions about the graph.
        """

        self._check_indexing(graph)
        graph = self._ensure_integrity(graph)

        return graph

    def _get_node_features(self, node):
        """
        Get the features of a node
        """
        if self.use_page_rank:
            neighbor_weights = nx.pagerank(self._graph, personalization={node: 1})
            normalized_neighbor_weights = {
                str(k): v for k, v in neighbor_weights.items()
            }

            return normalized_neighbor_weights

        if self.random_walk_length:
            neighbor_weights = {}

            for walk in range(self.number_of_walks):
                current = node

                for step in range(self.random_walk_length):
                    neighbors = [
                        neighbor for neighbor in self._graph.neighbors(current)
                    ]
                    current = random.choice(neighbors)

                    if current not in neighbor_weights:
                        neighbor_weights[current] = 1

                    else:
                        neighbor_weights[current] += 1

            # Convert keys to str fast
            normalized_neighbor_weights = {
                str(k): v for k, v in neighbor_weights.items()
            }
            self.neighborhoods[node] = normalized_neighbor_weights

            return normalized_neighbor_weights

        return [str(neighbor) for neighbor in self._graph.neighbors(node)]

    def generate_sketch(self):
        sketch = []

        for node in self._graph:
            simhash = CustomSimhash(self._get_node_features(node), f=self.dimensions)

            int_simhash = simhash.value
            binary_simhash = format(int_simhash, "b")

            # Pad with zeros
            if len(binary_simhash) < self.dimensions:
                binary_simhash = (
                    "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
                )

            self.hash_signatures[node] = simhash.sums

            sketch.append([int(bit) for bit in binary_simhash])

        self._sketch = sketch

    def _update_simple_simhash(self, edge):
        node1, node2 = edge

        # Update values for node1
        node1_hash = self.hash_signatures[node1]
        updated_simhash = CustomSimhash(
            [str(node2)], f=self.dimensions, sums=node1_hash
        )

        int_simhash = updated_simhash.value
        binary_simhash = format(int_simhash, "b")

        # Pad with zeros
        if len(binary_simhash) < self.dimensions:
            binary_simhash = (
                "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
            )

        self._sketch[node1] = [int(bit) for bit in binary_simhash]
        self.hash_signatures[node1] = updated_simhash.sums

        # Update values for node2
        node2_hash = self.hash_signatures[node2]
        updated_simhash = CustomSimhash(
            [str(node1)], f=self.dimensions, sums=node2_hash
        )

        # For random walk, we need to update the node features
        updated_simhash = CustomSimhash(
            self._get_node_features(node2), f=self.dimensions
        )

        int_simhash = updated_simhash.value
        binary_simhash = format(int_simhash, "b")

        # Pad with zeros
        if len(binary_simhash) < self.dimensions:
            binary_simhash = (
                "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
            )

        self._sketch[node2] = [int(bit) for bit in binary_simhash]
        self.hash_signatures[node2] = updated_simhash.sums

    def _update_page_rank(self, edge):
        node1, node2 = edge

        updated_simhash = CustomSimhash(
            self._get_node_features(node1), f=self.dimensions
        )
        int_simhash = updated_simhash.value
        binary_simhash = format(int_simhash, "b")

        # Pad with zeros
        if len(binary_simhash) < self.dimensions:
            binary_simhash = (
                "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
            )

        self._sketch[node2] = [int(bit) for bit in binary_simhash]

        updated_simhash = CustomSimhash(
            self._get_node_features(node2), f=self.dimensions
        )
        int_simhash = updated_simhash.value
        binary_simhash = format(int_simhash, "b")

        # Pad with zeros
        if len(binary_simhash) < self.dimensions:
            binary_simhash = (
                "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
            )

        self._sketch[node2] = [int(bit) for bit in binary_simhash]

    def _update_random_walk(self, edges):
        nodes_to_update = set()
        for edge in edges:
            node1, node2 = edge

            for node, neighborood in self.neighborhoods.items():
                if str(node1) in neighborood.keys() or str(node2) in neighborood.keys():
                    # Skip nodes that have a weight less than half of the number of walks
                    weight_limit = int(self.number_of_walks * 0.5)
                    if (
                        neighborood.get(str(node1), 0) < weight_limit
                        and neighborood.get(str(node2), 0) < weight_limit
                    ):
                        continue

                    nodes_to_update.add(node)

        print(f"Updating simhash for number of nodes: {len(nodes_to_update)}")
        for node in nodes_to_update:
            updated_simhash = CustomSimhash(
                self._get_node_features(node), f=self.dimensions
            )
            int_simhash = updated_simhash.value
            binary_simhash = format(int_simhash, "b")

            # Pad with zeros
            if len(binary_simhash) < self.dimensions:
                binary_simhash = (
                    "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
                )

            self._sketch[node] = [int(bit) for bit in binary_simhash]

        return

    def update_sketch(self, edges: list):
        """
        Update the sketch with the added edges
        """
        # Page rank case
        if self.use_page_rank:
            for edge in edges:
                self._update_page_rank()

            return

        # Random walk case
        if self.random_walk_length:
            self._update_random_walk(edges)

            return

        # Simple simhash case
        for edge in edges:
            self._update_simple_simhash(edge)

    def fit(self, graph: nx.Graph, added_edges: list = None):
        """
        Fit the Livesketch model on a given graph
        """
        self._set_seed()

        graph = self._check_graph(graph)
        self._graph = graph

        if not self._sketch:
            self.generate_sketch()

        if added_edges:
            self.update_sketch(added_edges)

    def get_embedding(self):
        """
        Generate the embedding from an already fitted model
        """
        if not self._sketch:
            raise ValueError("Model has not been fitted yet")

        embedding = np.array(self._sketch)

        return embedding
