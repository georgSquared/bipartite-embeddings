from karateclub.node_embedding.neighbourhood import nodesketch
from sklearn.linear_model import LogisticRegressionCV

from bipartite_embeddings.constants import EdgeOperator
from bipartite_embeddings.evaluator import Evaluator
from bipartite_embeddings.idioglossia import Idioglossia
from utils import load_graph


def main():
    graph = load_graph()

    # TODO: Test this
    #   - use multiple models
    #   - adjust classifier params (?)
    evaluator = Evaluator(
        graph,
        Idioglossia(dimensions=32),
        LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1000),
        EdgeOperator.CONCAT,
    )

    print(f"ROC AUC score: {evaluator.get_roc_auc_score()}")


if __name__ == "__main__":
    main()
