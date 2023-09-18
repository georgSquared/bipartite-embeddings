import networkx as nx

from sklearn.linear_model import LogisticRegressionCV
from stellargraph.data import EdgeSplitter
from livesketch import Livesketch
from utils import Datasets, load_graph

from bipartite_embeddings.constants import EdgeOperator, SimilarityMeasure
from bipartite_embeddings.evaluator import EmbeddingModel, Evaluator

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


def streamify(graph: nx.Graph, batch_count: int = None):
    """
    Simulate a stream of edges by splitting the graph into a base 50% starting graph and
    a stream of 50% edges.

    If a batch count is provided, the graph is evaluated after every batch of edges
    has been added to the base graph.

    The edges are added one by one to the base graph
    """
    reduced_graph, stream_edge_ids, stream_edge_labels = EdgeSplitter(
        graph
    ).train_test_split(p=0.5, method="global", keep_connected=True)

    added_edge_count = 0
    for idx, edge in enumerate(stream_edge_ids):
        added_edge_count += 1

        if stream_edge_labels[idx] == 1:
            reduced_graph.add_edge(edge[0], edge[1])
        else:
            continue

        if batch_count:
            if added_edge_count % batch_count == 0:
                evaluator = Evaluator(
                    reduced_graph,
                    Livesketch(dimensions=32),
                    classifier=LogisticRegressionCV(
                        Cs=10, cv=10, scoring="roc_auc", max_iter=10000
                    ),
                    embedding_operator=EdgeOperator.CONCAT,
                )

                print(f"Graph edge count: {reduced_graph.number_of_edges()}.")
                print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")

            continue

        else:
            evaluator = Evaluator(
                reduced_graph,
                Livesketch(dimensions=32),
                classifier=LogisticRegressionCV(
                    Cs=10, cv=10, scoring="roc_auc", max_iter=10000
                ),
                embedding_operator=EdgeOperator.CONCAT,
            )

            print(f"Graph edge count: {reduced_graph.number_of_edges()}")

            print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")


def main():
    graph = load_graph(dataset=Datasets.BLOG)

    models = [
        Livesketch(dimensions=32),
        NodeSketch(dimensions=32, decay=0.4, iterations=4),
        Node2Vec(),
    ]

    # roc_auc(graph)
    # streamify(graph, batch_count=100)

    results = 0
    for i in range(10):
        precision = precision_at_100(
            graph,
            embedding_model=models[0],
            similarity_measure=SimilarityMeasure.HAMMING,
        )

        print(f"Iteration: {i}. Precision@100: {precision}")
        results += precision

    print(f"Average precision@100: {results / 10}")


if __name__ == "__main__":
    main()
