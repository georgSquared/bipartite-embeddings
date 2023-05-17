from sklearn.linear_model import LogisticRegressionCV

from bipartite_embeddings.constants import EdgeOperator
from bipartite_embeddings.evaluator import Evaluator
from bipartite_embeddings.idioglossia import Idioglossia
from utils import load_graph


def roc_auc():
    graph = load_graph()

    evaluator = Evaluator(
        graph,
        Idioglossia(dimensions=128, decay=0.2, iterations=4),
        classifier=LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc"),
        embedding_operator=EdgeOperator.AVERAGE,
    )

    print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")


def precision_at_100():
    graph = load_graph()

    evaluator = Evaluator(
        graph,
        Idioglossia(dimensions=128, decay=0.2, iterations=4),
    )

    print(f"Precsion@100: {evaluator.get_precision_at_100()}")


def main():
    roc_auc()
    precision_at_100()


if __name__ == "__main__":
    main()
