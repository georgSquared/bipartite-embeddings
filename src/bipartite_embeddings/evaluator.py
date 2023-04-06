from typing import Protocol, List

import networkx as nx
import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from bipartite_embeddings.constants import SampleType, EdgeOperator
from bipartite_embeddings.utils import get_train_test_samples, DotDict


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


class Evaluator:
    def __init__(
        self,
        graph: nx.Graph,
        embedding_model: EmbeddingModel,
        classifier: ClassifierEstimator,
        embedding_operator: EdgeOperator,
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.embedding_operator = embedding_operator

    def get_samples(self) -> DotDict:
        return get_train_test_samples(self.graph)

    @property
    def samples(self):
        return self.get_samples()

    def get_node_embeddings(self, sample_type: SampleType) -> ndarray:
        self.embedding_model.fit(
            self.samples.G_train
            if sample_type == SampleType.TRAIN
            else self.samples.G_test
        )

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

        elif self.embedding_operator == EdgeOperator.COSINE:
            return [
                cosine_similarity(node_embeddings[edge[0]], node_embeddings[edge[1]])
                for edge in samples
            ]

        raise ValueError(f"Unknown edge operator: {self.embedding_operator}")

    def get_roc_auc_score(self) -> float:
        # Get the edge embeddings for the train Graph, and fit the classifier
        train_edge_embeddings = self.get_edge_embeddings(sample_type=SampleType.TRAIN)
        self.classifier.fit(train_edge_embeddings, self.samples.edge_labels_train)

        # Get the edge embeddings for the test Graph, and predict the labels of the test data
        test_edge_embeddings = self.get_edge_embeddings(sample_type=SampleType.TEST)
        predictions = self.classifier.predict_proba(test_edge_embeddings)[:, 1]

        # Compute the ROC AUC score
        score = roc_auc_score(self.samples.edge_labels_test, predictions)

        return score
