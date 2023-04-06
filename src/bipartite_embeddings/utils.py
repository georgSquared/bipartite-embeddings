import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import networkx as nx
from sklearn.model_selection import train_test_split

from stellargraph.data import EdgeSplitter


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )


def get_root_dir():
    return Path(__file__).parent.parent.parent


@contextmanager
def performance_measuring() -> float:
    start = perf_counter()
    try:
        yield
    finally:
        print(f"Took: {perf_counter() - start} seconds")


def load_graph(edgelist_path=None):
    """
    Load the graph from a given edgelist file
    :return:
    """
    if not edgelist_path:
        edgelist_path = os.path.join(
            get_root_dir(), "data", "ml-latest-small", "ratings_edgelist.csv"
        )

    graph = nx.read_edgelist(edgelist_path, delimiter=",")

    # Relabel the nodes to have sequential integer labels
    sequential_graph = nx.convert_node_labels_to_integers(
        graph, label_attribute="original_id"
    )

    return sequential_graph


def get_train_test_samples(G: nx.Graph) -> DotDict:
    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed.
    # In short: G - test edges = G_test
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test,
    # and obtain the reduced graph G_train with the sampled links removed
    # In short: G_test - train edges = G_train
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    # Split the train edges into train and model selection
    # examples_model_selection are used for parameter tuning and to essentialy train the classifier
    # think of them as classifier test data
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(
        edge_ids_train, edge_labels_train, train_size=0.75, test_size=0.25
    )

    return DotDict(
        G_train=G_train,
        edge_ids_train=edge_ids_train,
        edge_labels_train=edge_labels_train,
        G_test=G_test,
        edge_ids_test=edge_ids_test,
        edge_labels_test=edge_labels_test,
        examples_train=examples_train,
        examples_model_selection=examples_model_selection,
        labels_train=labels_train,
        labels_model_selection=labels_model_selection,
    )
