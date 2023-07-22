import networkx as nx

from livesketch import LiveSketch
from sklearn.linear_model import LogisticRegressionCV
from stellargraph.data import EdgeSplitter
from utils import Datasets, load_graph

from bipartite_embeddings.constants import EdgeOperator
from bipartite_embeddings.evaluator import Evaluator


def roc_auc(graph: nx.Graph):
    evaluator = Evaluator(
        graph,
        LiveSketch(dimensions=128, decay=0.2, iterations=4),
        classifier=LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc"),
        embedding_operator=EdgeOperator.CONCAT,
    )

    print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")


def precision_at_100(graph: nx.Graph):
    evaluator = Evaluator(
        graph,
        LiveSketch(dimensions=128, decay=0.2, iterations=1),
    )

    # evaluator = Evaluator(
    #     graph,
    #     NodeSketch(dimensions=32, decay=0.1, iterations=2),
    # )

    # evaluator = Evaluator(graph, Node2Vec())

    print(f"Precsion@100: {evaluator.get_precision_at_100()}")


def streamify(graph: nx.Graph, batch_count: int = None):
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
                    LiveSketch(dimensions=32),
                    classifier=LogisticRegressionCV(
                        Cs=10, cv=10, scoring="roc_auc", max_iter=10000
                    ),
                    embedding_operator=EdgeOperator.CONCAT,
                )

                print(
                    f"Graph edge count: {reduced_graph.number_of_edges()}. ROC AUC score: {evaluator.get_roc_auc_score()}"
                )

            continue

        else:
            evaluator = Evaluator(
                reduced_graph,
                LiveSketch(dimensions=32),
                classifier=LogisticRegressionCV(
                    Cs=10, cv=10, scoring="roc_auc", max_iter=10000
                ),
                embedding_operator=EdgeOperator.CONCAT,
            )

            print(
                f"Graph edge count: {reduced_graph.number_of_edges()}. ROC AUC score: {evaluator.get_roc_auc_score()}"
            )


def main():
    graph = load_graph(dataset=Datasets.PPI)

    # roc_auc(graph)
    precision_at_100(graph)

    # streamify(graph, batch_count=100)


if __name__ == "__main__":
    main()
