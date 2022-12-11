import os
from pathlib import Path

import networkx as nx


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
