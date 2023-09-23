from collections import defaultdict
import typing
from typing import Protocol, List

import networkx as nx
import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from bipartite_embeddings.constants import SampleType, EdgeOperator, SimilarityMeasure
from bipartite_embeddings.utils import (
    get_first_100_edges,
    get_train_test_samples,
    DotDict,
    performance_measuring,
)

from stellargraph.data import EdgeSplitter


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
        elif similarity_measure == SimilarityMeasure.HAMMING:
            # This calculates distances instead of similarities
            similarities = (embeddings[:, None, :] == embeddings).sum(2)
        elif similarity_measure == SimilarityMeasure.DOT_PRODUCT:
            similarities = np.dot(embeddings, embeddings.T)

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
        with performance_measuring(message="Node embeddings calculation"):
            node_embeddings = self.get_node_embeddings(sample_type=SampleType.TRAIN)

        # Get the similarity matrix
        similarities = self.get_similarity_matrix(node_embeddings, similarity_measure)

        return get_first_100_edges(similarities, self.graph, self.samples.G_train)

        # # Check with the test edges
        # top_similarities = []
        # for edge in self.samples.edge_ids_test:
        #     if edge[0] == edge[1] or edge[0] > edge[1]:
        #         continue

        #     edge_similarity = similarities[edge[0], edge[1]]
        #     top_similarities.append((edge[0], edge[1], edge_similarity))

        # top_similarities = sorted(top_similarities, key=lambda x: x[2], reverse=True)

        # tp = 0
        # for edge in top_similarities[:100]:
        #     if self.graph.has_edge(edge[0], edge[1]):
        #         tp += 1

        # return tp / 100


class StreamingEvaluator:
    def __init__(
        self,
        graph: nx.Graph,
        embedding_model: EmbeddingModel,
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        self.samples = get_train_test_samples(self.graph)

        reduced_graph, stream_edge_ids, stream_edge_labels = EdgeSplitter(
            self.samples.G_train
        ).train_test_split(p=0.5, method="global", keep_connected=True)

        self.reduced_graph = reduced_graph
        self.stream_edge_ids = stream_edge_ids
        self.stream_edge_labels = stream_edge_labels
        self.updated_neighbours = defaultdict(list)

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

    def get_node_embeddings(self, updated_neighbours: dict = None) -> ndarray:
        try:
            self.embedding_model.fit(
                self.samples.G_train, updated_neighbours=updated_neighbours
            )
        except TypeError:
            print("Model does not support updates. Fitting from scratch")
            self.embedding_model.fit(self.samples.G_train)

        return self.embedding_model.get_embedding()

    def get_precision_at_100(
        self,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.HAMMING,
        updated_neighbours: dict = None,
    ) -> float:
        # Get the node embeddings for the train Graph
        with performance_measuring(message="Node embeddings calculation"):
            node_embeddings = self.get_node_embeddings(
                updated_neighbours=updated_neighbours
            )

        with performance_measuring(message="Metric calculation"):
            # Get the similarity matrix
            similarities = self.get_similarity_matrix(
                node_embeddings, similarity_measure
            )

            return get_first_100_edges(similarities, self.graph, self.samples.G_train)

            # # Check with the top 100 similarity scores of non-edges
            # top_non_edges = get_top_100_non_edges(similarities, self.samples.G_train)
            # tp = 0
            # for edge in top_non_edges:
            #     if self.graph.has_edge(edge[0], edge[1]):
            #         tp += 1

            # return tp / 100

            # Check with the test edges
            top_similarities = []
            for edge in self.samples.edge_ids_test:
                if edge[0] == edge[1] or edge[0] > edge[1]:
                    continue

                edge_similarity = similarities[edge[0], edge[1]]
                top_similarities.append((edge[0], edge[1], edge_similarity))

            top_similarities = sorted(
                top_similarities, key=lambda x: x[2], reverse=True
            )

            tp = 0
            for edge in top_similarities[:100]:
                if self.graph.has_edge(edge[0], edge[1]):
                    tp += 1

            return tp / 100

    def streamify(self, batch_count: int = None):
        print(f"Inital precision@100: {self.get_precision_at_100()} \n")
        print("Starting stream \n")

        added_edge_count = 0
        for edge in self.stream_edge_ids:
            if self.samples.G_train.has_edge(edge[0], edge[1]):
                continue

            added_edge_count += 1
            self.samples.G_train.add_edge(edge[0], edge[1])
            self.updated_neighbours[edge[0]].append(edge[1])

            if batch_count:
                if added_edge_count % batch_count == 0:
                    precision = self.get_precision_at_100(
                        updated_neighbours=self.updated_neighbours,
                    )

                    print(
                        f"Graph edge count: {self.samples.G_train.number_of_edges()}."
                    )
                    print(f"Precision@100: {precision}")
                    print()

                    # Reset the updated neighbours
                    self.updated_neighbours = defaultdict(list)

                continue

            else:
                precision = self.get_precision_at_100(
                    updated_neighbours=self.updated_neighbours,
                )

                print(f"Graph edge count: {self.samples.G_train.number_of_edges()}.")
                print(f"Precision@100: {precision}")
                print()

                # Reset the updated neighbours
                self.updated_neighbours = defaultdict(list)
