"""
This module contains our custom embedding model

The name is a placeholder
"""
import json
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
        use_page_rank: bool = False,
    ):
        self.dimensions = dimensions
        self._sketch = None
        self.seed = seed
        self.hash_signatures = {}

        if random_walk_length and use_page_rank:
            raise ValueError("You can only use one of random walk length and page rank")

        self.random_walk_length = random_walk_length
        self.number_of_walks = 1000
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
            # json.loads(json.dumps()) is used to convert keys to str fast
            neighbor_weights = json.loads(
                json.dumps(nx.pagerank(self._graph, personalization={node: 1}))
            )

            return neighbor_weights

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

            # json.loads(json.dumps()) is used to convert keys to str fast
            return json.loads(json.dumps(neighbor_weights))

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

    def update_sketch(self, edge):
        """
        Update the sketch with a new edge
        """
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

        int_simhash = updated_simhash.value
        binary_simhash = format(int_simhash, "b")

        # Pad with zeros
        if len(binary_simhash) < self.dimensions:
            binary_simhash = (
                "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
            )

        self._sketch[node2] = [int(bit) for bit in binary_simhash]
        self.hash_signatures[node2] = updated_simhash.sums

    def fit(self, graph: nx.Graph, updated_neighbours: dict = None):
        """
        Fit the Livesketch model on a given graph
        """
        self._set_seed()

        graph = self._check_graph(graph)
        self._graph = graph

        if not self._sketch:
            self.generate_sketch()

        if not updated_neighbours:
            return

        # Update the sketch with the new edges
        for node, neighbours in updated_neighbours.items():
            for neighbour in neighbours:
                self.update_sketch((node, neighbour))

    def get_embedding(self):
        """
        Generate the embedding from an already fitted model
        """
        if not self._sketch:
            raise ValueError("Model has not been fitted yet")

        embedding = np.array(self._sketch)

        return embedding
