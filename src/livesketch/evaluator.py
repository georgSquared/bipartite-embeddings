import typing
from typing import List, Protocol

import networkx as nx
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph.data import EdgeSplitter

from livesketch.constants import EdgeOperator, SampleType, SimilarityMeasure
from livesketch.utils import (
    DotDict,
    get_first_100_edges_precision,
    get_train_test_samples,
    performance_measuring,
    top_hamming_similarity_chunked,
)


class EmbeddingModel(Protocol):
    def fit(self, g: nx.Graph):
        pass

    def get_embedding(self) -> ndarray:
        pass


class ClassifierEstimator(Protocol):
    def fit(self, x: List[ndarray], y: ndarray):
        pass

    def predict(self, x: List[ndarray]) -> ndarray:
        pass

    def predict_proba(self, x: List[ndarray]) -> ndarray:
        pass


# TODO: Add performance measuring in the correct spots


class Evaluator:
    def __init__(
        self,
        graph: nx.Graph,
        embedding_model: EmbeddingModel,
        classifier: typing.Optional[ClassifierEstimator] = None,
        embedding_operator: typing.Optional[EdgeOperator] = None,
    ):
        self.graph = graph
        self.large_graph = graph.number_of_nodes() > 100_000

        self.embedding_model = embedding_model
        self.classifier = classifier
        self.embedding_operator = embedding_operator
        self._samples = None

    def get_samples(self) -> DotDict:
        return get_train_test_samples(self.graph)

    @property
    def samples(self):
        if not self._samples:
            self._samples = self.get_samples()

        return self._samples

    def get_node_embeddings(
        self, sample_type: typing.Optional[SampleType] = None
    ) -> ndarray:
        if sample_type == SampleType.TRAIN:
            self.embedding_model.fit(self.samples.G_train)

        elif sample_type == SampleType.TEST:
            self.embedding_model.fit(self.samples.G_test)

        else:
            self.embedding_model.fit(self.graph)

        return self.embedding_model.get_embedding()

    @property
    def train_node_embeddings(self) -> ndarray:
        return self.get_node_embeddings(sample_type=SampleType.TRAIN)

    @property
    def test_node_embeddings(self) -> ndarray:
        return self.get_node_embeddings(sample_type=SampleType.TEST)

    def get_edge_embeddings(self, sample_type: SampleType) -> List[ndarray]:
        node_embeddings = (
            self.train_node_embeddings
            if sample_type == SampleType.TRAIN
            else self.test_node_embeddings
        )
        samples = (
            self.samples.edge_ids_train
            if sample_type == SampleType.TRAIN
            else self.samples.edge_ids_test
        )

        if self.embedding_operator == EdgeOperator.CONCAT:
            return [
                np.concatenate((node_embeddings[edge[0]], node_embeddings[edge[1]]))
                for edge in samples
            ]

        # This translates to using different node operations
        # to compute the edge embeddings
        # i.e. Cosine Similarity, Hadamard Product, Jaccard Distance, etc.
        elif self.embedding_operator == EdgeOperator.HADAMARD:
            return [
                np.multiply(node_embeddings[edge[0]], node_embeddings[edge[1]])
                for edge in samples
            ]

        elif self.embedding_operator == EdgeOperator.AVERAGE:
            return [
                (node_embeddings[edge[0]] + node_embeddings[edge[1]]) / 2
                for edge in samples
            ]

        raise ValueError(f"Unknown edge operator: {self.embedding_operator}")

    def get_similarity_matrix(
        self, embeddings: ndarray, similarity_measure: SimilarityMeasure
    ) -> ndarray:
        if similarity_measure == SimilarityMeasure.COSINE:
            similarities = cosine_similarity(embeddings)
        elif similarity_measure == SimilarityMeasure.DOT_PRODUCT:
            similarities = np.dot(embeddings, embeddings.T)
        elif similarity_measure == SimilarityMeasure.HAMMING:
            similarities = (embeddings[:, None, :] == embeddings).sum(2)

        else:
            raise ValueError(f"Unknown similarity measure: {similarity_measure}")

        return similarities

    def get_roc_auc_score(self) -> float:
        if not self.classifier or not self.embedding_operator:
            raise ValueError(
                "ROC AUC score can only be computed when classifier and embedding operator are both set"  # noqa: E501
            )

        # Get the edge embeddings for the train Graph, and fit the classifier
        train_edge_embeddings = self.get_edge_embeddings(sample_type=SampleType.TRAIN)
        self.classifier.fit(train_edge_embeddings, self.samples.edge_labels_train)

        # Get the edge embeddings for the test Graph,
        # and predict the labels of the test data
        test_edge_embeddings = self.get_edge_embeddings(sample_type=SampleType.TEST)
        predictions = self.classifier.predict_proba(test_edge_embeddings)[:, 1]

        # Compute the ROC AUC score
        score = roc_auc_score(self.samples.edge_labels_test, predictions)

        return score

    def get_precision_at_100(
        self, similarity_measure: SimilarityMeasure = SimilarityMeasure.HAMMING
    ) -> float:
        # Get the node embeddings for the train Graph
        with performance_measuring(message="Node embeddings calculation") as emb_perf:
            node_embeddings = self.get_node_embeddings(sample_type=SampleType.TRAIN)

        with performance_measuring(message="Metric calculation"):
            # If the graph is large, calculate the similarities and top similarities in chunks
            # Only hamming distance is supported for large graphs
            if self.large_graph:
                if similarity_measure != SimilarityMeasure.HAMMING:
                    raise ValueError(
                        "Large graphs only support hamming distance similarity measure"
                    )

                # Top similarities are of the form: (similarity, node1, node2)
                top_similarities = top_hamming_similarity_chunked(
                    csr_matrix(node_embeddings)
                )

                traversed_count = 0
                tp = 0

                for sim_tuple in top_similarities:
                    similiarity, node1, node2 = sim_tuple

                    # Only consider edges that are not in the train graph
                    if self.samples.G_train.has_edge(node1, node2):
                        continue

                    traversed_count += 1
                    if self.graph.has_edge(node1, node2):
                        tp += 1

                    if tp >= 100:
                        break

                precision = tp / traversed_count
                return precision, emb_perf.elapsed

            # Get the similarity matrix
            similarities = self.get_similarity_matrix(
                node_embeddings, similarity_measure
            )

            precision = get_first_100_edges_precision(
                similarities, self.graph, self.samples.G_train
            )
            return precision, emb_perf.elapsed


class StreamingEvaluator:
    def __init__(
        self,
        graph: nx.Graph,
        embedding_model: EmbeddingModel,
        initial_graph_percentage: float = 0.9,
    ):
        self.graph = graph
        self.embedding_model = embedding_model

        p_val = 1 - initial_graph_percentage
        reduced_graph, stream_edge_ids, stream_edge_labels = EdgeSplitter(
            self.graph
        ).train_test_split(p=p_val, method="global", keep_connected=True)

        self.reduced_graph = reduced_graph
        self.stream_edge_ids = stream_edge_ids
        self.stream_edge_labels = stream_edge_labels
        self.added_edges = []

    def get_similarity_matrix(
        self, embeddings: ndarray, similarity_measure: SimilarityMeasure
    ) -> ndarray:
        if similarity_measure == SimilarityMeasure.COSINE:
            similarities = cosine_similarity(embeddings)
        elif similarity_measure == SimilarityMeasure.HAMMING:
            # This calculates distances instead of similarities
            similarities = (embeddings[:, None, :] == embeddings).sum(2)
        elif similarity_measure == SimilarityMeasure.DOT_PRODUCT:
            similarities = np.dot(embeddings, embeddings.T)

        else:
            raise ValueError(f"Unknown similarity measure: {similarity_measure}")

        return similarities

    def get_node_embeddings(self, added_edges: list = None) -> ndarray:
        try:
            # Perhaps this should be the reduced graph?
            self.embedding_model.fit(self.reduced_graph, added_edges=added_edges)
        except TypeError as ex:
            print(ex)
            print("Model does not support updates. Fitting from scratch")
            self.embedding_model.fit(self.reduced_graph)

        return self.embedding_model.get_embedding()

    def get_precision_at_100(
        self,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.HAMMING,
        added_edges: list = None,
    ) -> float:
        # Get the node embeddings for the train Graph
        with performance_measuring(message="Node embeddings calculation"):
            node_embeddings = self.get_node_embeddings(added_edges=added_edges)

        with performance_measuring(message="Metric calculation"):
            # Get the similarity matrix
            similarities = self.get_similarity_matrix(
                node_embeddings, similarity_measure
            )

            return get_first_100_edges_precision(
                similarities, self.graph, self.reduced_graph
            )

    def streamify(self, batch_count: int = None):
        print(f"Inital precision@100: {self.get_precision_at_100()} \n")
        print("Starting stream \n")

        added_edge_count = 0
        for edge in self.stream_edge_ids:
            if self.reduced_graph.has_edge(edge[0], edge[1]):
                continue

            if edge[0] >= edge[1]:
                continue

            added_edge_count += 1
            self.reduced_graph.add_edge(edge[0], edge[1])
            self.added_edges.append(edge)

            if batch_count:
                if added_edge_count % batch_count == 0:
                    precision = self.get_precision_at_100(added_edges=self.added_edges)

                    print(f"Graph edge count: {self.reduced_graph.number_of_edges()}.")
                    print(f"Precision@100: {precision}")
                    print()

                    # Reset the added edges
                    self.added_edges = []

                continue

            else:
                precision = self.get_precision_at_100(added_edges=self.added_edges)

                print(f"Graph edge count: {self.reduced_graph.number_of_edges()}.")
                print(f"Precision@100: {precision}")
                print()

                # Reset the added edges
                self.added_edges = []
