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


def run_full_graph(
    graph: nx.Graph, model_class: EmbeddingModel, model_args: dict, iterations=10
):
    results = 0
    for i in range(iterations):
        model = model_class(**model_args)

        evaluator = Evaluator(graph, model)
        pr100 = evaluator.get_precision_at_100(
            similarity_measure=SimilarityMeasure.HAMMING,
        )

        print(f"Iteration: {i}. Precision@100: {pr100}")
        results += pr100

    print(f"Average precision@100: {results / iterations}")


def main():
    graph = load_graph(dataset=Datasets.DBLP)

    models = [
        (Livesketch, dict(dimensions=128)),
        (NodeSketch, dict(dimensions=128, decay=0.4, iterations=2)),
        (Node2Vec, dict()),
        (Livesketch, dict(dimensions=128, use_page_rank=True)),
        (
            Livesketch,
            dict(dimensions=128, random_walk_length=3, number_of_walks=500),
        ),
    ]
    model_index = 4

    run_full_graph(graph, models[model_index][0], models[model_index][1], iterations=1)

    # model = models[model_index][0](**models[model_index][1])
    # streamify(graph, model=model, batch_count=1000)


if __name__ == "__main__":
    main()
