"""
This module contains our custom embedding model

The name is a placeholder
"""
import random

import networkx as nx
import numpy as np
from simhash import Simhash


class Livesketch:
    """
    A simhash based edge embedding model
    """

    def __init__(
        self,
        dimensions: int = 32,
        seed: int = 42,
    ):
        self.dimensions = dimensions
        self._sketch = None
        self.seed = seed

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
        return [str(neighbor) for neighbor in self._graph.neighbors(node)]

    def generate_sketch(self):
        sketch = []

        for node in self._graph:
            int_simhash = Simhash(
                self._get_node_features(node), f=self.dimensions
            ).value

            binary_simhash = format(int_simhash, "b")

            # Pad with zeros
            if len(binary_simhash) < self.dimensions:
                binary_simhash = (
                    "0" * (self.dimensions - len(binary_simhash)) + binary_simhash
                )

            sketch.append([int(bit) for bit in binary_simhash])

        self._sketch = sketch

    def fit(self, graph):
        """
        Fit the Livesketch model on a given graph
        """
        self._set_seed()

        graph = self._check_graph(graph)
        self._graph = graph

        self.generate_sketch()

    def get_embedding(self):
        """
        Generate the embedding from an already fitted model
        """
        if not self._sketch:
            raise ValueError("Model has not been fitted yet")

        embedding = np.array(self._sketch)

        return embedding
