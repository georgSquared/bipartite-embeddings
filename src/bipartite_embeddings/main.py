import networkx as nx

from sklearn.linear_model import LogisticRegressionCV
from livesketch import Livesketch
from utils import Datasets, load_graph

from bipartite_embeddings.constants import EdgeOperator, SimilarityMeasure
from bipartite_embeddings.evaluator import EmbeddingModel, Evaluator, StreamingEvaluator

from karateclub.node_embedding.neighbourhood.nodesketch import NodeSketch
from karateclub.node_embedding.neighbourhood.node2vec import Node2Vec


def roc_auc(graph: nx.Graph):
    evaluator = Evaluator(
        graph,
        Livesketch(dimensions=128, decay=0.2, iterations=4),
        classifier=LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc"),
        embedding_operator=EdgeOperator.CONCAT,
    )

    print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")


def precision_at_100(
    graph: nx.Graph,
    embedding_model: EmbeddingModel,
    similarity_measure: SimilarityMeasure,
):
    """
    Calculate the precision at 100 for a given graph and embedding model

    Similarities are calculated using the given similarity measure
    """
    evaluator = Evaluator(
        graph,
        embedding_model,
    )
    pr100 = evaluator.get_precision_at_100(similarity_measure=similarity_measure)

    return pr100


def streamify(graph: nx.Graph, model: EmbeddingModel, batch_count: int = None):
    """
    Simulate a stream of edges by splitting the graph into a base 50% starting graph and
    a stream of 50% edges.

    If a batch count is provided, the graph is evaluated after every batch of edges
    has been added to the base graph.

    The edges are added one by one to the base graph
    """
    evaluator = StreamingEvaluator(
        graph=graph,
        embedding_model=model,
    )

    evaluator.streamify(batch_count=batch_count)


def run_full_graph(graph: nx.Graph, model: EmbeddingModel, iterations=10):
    results = 0
    for i in range(iterations):
        precision = precision_at_100(
            graph,
            embedding_model=model,
            similarity_measure=SimilarityMeasure.HAMMING,
        )

        print(f"Iteration: {i}. Precision@100: {precision}")
        results += precision

    print(f"Average precision@100: {results / 10}")


def main():
    graph = load_graph(dataset=Datasets.PPI)

    models = [
        Livesketch(dimensions=32),
        NodeSketch(dimensions=32, decay=0.4, iterations=2),
        Node2Vec(),
        Livesketch(dimensions=32, use_page_rank=True),
        Livesketch(dimensions=32, random_walk_length=3),
    ]

    # TODO: Fix iterations, they reuse the same model and it auto-updates
    # run_full_graph(graph, models[4], iterations=10)

    streamify(graph, model=models[4], batch_count=1000)


if __name__ == "__main__":
    main()
