import argparse
import datetime
import os
import networkx as nx
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from livesketch import Livesketch
from utils import Datasets, load_graph

from bipartite_embeddings.constants import ROOT_DIR, EdgeOperator, SimilarityMeasure
from bipartite_embeddings.evaluator import EmbeddingModel, Evaluator, StreamingEvaluator

from karateclub.node_embedding.neighbourhood.nodesketch import NodeSketch


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
    Simulate a stream of edges by splitting the graph into a base percentage% starting graph and
    a stream of remaining edges.

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
    graph: nx.Graph,
    model_class: EmbeddingModel.__class__,
    model_args: dict,
    iterations=10,
    results_dict=None,
    dataset=None,
):
    results = 0
    for i in range(iterations):
        model = model_class(**model_args)

        evaluator = Evaluator(graph, model)
        pr100, elapsed = evaluator.get_precision_at_100(
            similarity_measure=SimilarityMeasure.HAMMING,
        )

        print(f"Iteration: {i}. Precision@100: {pr100}")
        results += pr100

        results_dict["dataset"].append(dataset)
        results_dict["algorithm"].append(model_class.__name__)
        results_dict["dimensions"].append(model_args["dimensions"])
        results_dict["random_walk_length"].append(
            model_args.get("random_walk_length", 0)
        )
        results_dict["number_of_walks"].append(model_args.get("number_of_walks", 0))
        results_dict["iteration"].append(i)
        results_dict["precision"].append(pr100)
        results_dict["run_time"].append(elapsed)

    print(f"Average precision@100: {results / iterations}")

    return results_dict


def run_experiments():
    for dataset in [Datasets.BLOG]:
        print(f"Running experiment for {dataset.value}")
        graph = load_graph(dataset=dataset)

        results_dict = {
            "dataset": [],
            "algorithm": [],
            "dimensions": [],
            "random_walk_length": [],
            "number_of_walks": [],
            "precision": [],
            "run_time": [],
            "iteration": [],
        }

        models = [
            (Livesketch, dict(dimensions=128)),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=2, number_of_walks=500),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=3, number_of_walks=500),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=4, number_of_walks=500),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=5, number_of_walks=500),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=3, number_of_walks=100),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=3, number_of_walks=200),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=3, number_of_walks=500),
            ),
            (
                Livesketch,
                dict(dimensions=128, random_walk_length=3, number_of_walks=1000),
            ),
            (NodeSketch, dict(dimensions=128, decay=0.2, iterations=4)),
        ]

        for model_idx in range(len(models)):
            results_dict = run_full_graph(
                graph,
                models[model_idx][0],
                models[model_idx][1],
                iterations=1,
                dataset=dataset.value,
                results_dict=results_dict,
            )

        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(
            os.path.join(
                ROOT_DIR,
                "data",
                "results",
                f"{dataset.value} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="streaming or default", default="default")
    parser.add_argument(
        "--dataset",
        help="Dataset to run experiments on. Only compatible with streaming mode",
        default=Datasets.BLOG.value,
    )

    args = parser.parse_args()
    if args.mode == "default":
        run_experiments()

    elif args.mode == "streaming":
        graph = load_graph(dataset=Datasets(args.dataset))

        streamify(
            graph,
            Livesketch(dimensions=128, random_walk_length=3, number_of_walks=500),
            batch_count=1000,  # Calculate precision after 10% of edges
        )

    else:
        print("Invalid mode")

    return


if __name__ == "__main__":
    main()
