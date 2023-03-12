import os
from pathlib import Path

import networkx as nx
from sklearn.model_selection import train_test_split

from stellargraph.data import EdgeSplitter


def get_root_dir():
    return Path(__file__).parent.parent.parent


def load_graph(edgelist_path=None):
    """
    Load the graph from a given edgelist file
    :return:
    """
    if not edgelist_path:
        edgelist_path = os.path.join(
            get_root_dir(), "data", "ml-latest-small", "ratings_edgelist.csv"
        )

    return nx.read_edgelist(edgelist_path, delimiter=",")


def get_train_test_samples(G):
    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test,
    # and obtain the reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(
        edge_ids_train, edge_labels_train, train_size=0.75, test_size=0.25
    )

    return {
        "G_train": G_train,
        "edge_ids_train": edge_ids_train,
        "edge_labels_train": edge_labels_train,
        "G_test": G_test,
        "edge_ids_test": edge_ids_test,
        "edge_labels_test": edge_labels_test,
    }
