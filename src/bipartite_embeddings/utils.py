import heapq
import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from stellargraph.data import EdgeSplitter
from scipy.sparse import csr_matrix
from tqdm import tqdm


class Datasets(Enum):
    ML_SMALL = "ml-small"
    BLOG = "blog"
    PPI = "ppi"
    WIKI = "wiki"
    DBLP = "dblp"
    YOUTUBE = "youtube"
    LIVEJOURNAL = "livejournal"


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
def performance_measuring(message: str = None) -> float:
    start = perf_counter()
    try:
        yield
    finally:
        print(f"{message or 'Operation'}: Took: {perf_counter() - start} seconds")


def load_graph(edgelist_path=None, dataset: Datasets = Datasets.ML_SMALL):
    """
    Load the graph from a given edgelist file
    :return:
    """

    if not edgelist_path:
        if dataset == Datasets.ML_SMALL:
            edgelist_path = os.path.join(
                get_root_dir(), "data", "ml-latest-small", "ratings_edgelist.csv"
            )
        elif dataset == Datasets.BLOG:
            edgelist_path = os.path.join(get_root_dir(), "data", "blog", "edges.csv")
        elif dataset == Datasets.PPI:
            edgelist_path = os.path.join(get_root_dir(), "data", "ppi", "edges.csv")
        elif dataset == Datasets.WIKI:
            edgelist_path = os.path.join(get_root_dir(), "data", "wiki", "edges.csv")
        elif dataset == Datasets.DBLP:
            edgelist_path = os.path.join(get_root_dir(), "data", "dblp", "edges.csv")
        elif dataset == Datasets.YOUTUBE:
            edgelist_path = os.path.join(get_root_dir(), "data", "youtube", "edges.csv")
        elif dataset == Datasets.LIVEJOURNAL:
            edgelist_path = os.path.join(
                get_root_dir(), "data", "livejournal", "edges.csv"
            )
        else:
            raise ValueError("Dataset not supported")

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
    # examples_model_selection are used for parameter tuning and to essentially train the classifier
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


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_top_100_non_edges(similarities, graph):
    # Define a mask to filter out the upper triangular matrix
    # (including the diagonal)
    similarities = np.tril(similarities, k=-1)

    # 100000 Arbitrary number of top indices. Should adjust dynamically
    indices = np.argpartition(similarities.ravel(), -100000)[-100000:]
    indices = indices[np.argsort(similarities.ravel()[indices])][::-1]
    top_indices = np.unravel_index(indices, similarities.shape)

    top_indices = list(zip(top_indices[0], top_indices[1]))

    top_non_edges = []
    for idx in top_indices:
        if graph.has_edge(idx[0], idx[1]):
            continue

        top_non_edges.append(idx)

        if len(top_non_edges) >= 100:
            break

    if len(top_non_edges) < 100:
        raise ValueError("Not enough non edges")

    return top_non_edges


# Bad name, it returns precision not edges
def get_first_100_edges_precision(similarities, original_graph, train_graph):
    # Define a mask to filter out the upper triangular matrix
    # (including the diagonal)
    similarities = np.tril(similarities, k=-1)

    # 100000 Arbitrary number of top indices. Should adjust dynamically
    indices = np.argpartition(similarities.ravel(), -100000)[-100000:]
    indices = indices[np.argsort(similarities.ravel()[indices])][::-1]
    top_indices = np.unravel_index(indices, similarities.shape)

    top_indices = list(zip(top_indices[0], top_indices[1]))

    traversed_count = 0
    tp = 0

    for idx in top_indices:
        # Only consider edges that are not in the train graph
        if train_graph.has_edge(idx[0], idx[1]):
            continue

        traversed_count += 1
        if original_graph.has_edge(idx[0], idx[1]):
            tp += 1

        if tp >= 100:
            break

    return tp / traversed_count


def get_test_edges_precision(similarities, test_edges, graph):
    # Check with the test edges
    top_similarities = []
    for edge in test_edges:
        if edge[0] == edge[1] or edge[0] > edge[1]:
            continue

        edge_similarity = similarities[edge[0], edge[1]]
        top_similarities.append((edge[0], edge[1], edge_similarity))

    top_similarities = sorted(top_similarities, key=lambda x: x[2], reverse=True)

    tp = 0
    for edge in top_similarities[:100]:
        if graph.has_edge(edge[0], edge[1]):
            tp += 1

    return tp / 100


def get_top_similarity_precision(similarities, original_graph, train_graph):
    # Check with the top 100 similarity scores of non-edges
    top_non_edges = get_top_100_non_edges(similarities, train_graph)
    tp = 0
    for edge in top_non_edges:
        if original_graph.has_edge(edge[0], edge[1]):
            tp += 1

    return tp / 100


def pairwise_hamming_distance_chunked(
    X: csr_matrix, chunk_size: int = 10000
) -> np.ndarray:
    """Compute pairwise hamming distance for a sparse matrix using a chunking approach."""

    n = X.shape[0]
    result = np.zeros((n, n), dtype=np.int32)

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)

            dot_product = X[i:end_i] @ X[j:end_j].T
            dot_product = dot_product.todense()

            hamming_dist = X.shape[1] - dot_product

            result[i:end_i, j:end_j] = hamming_dist

    return result


def pairwise_hamming_similarity_chunked(
    X: csr_matrix, chunk_size: int = 1000
) -> np.ndarray:
    """Compute pairwise hamming similarity for a sparse matrix using a chunking approach."""

    n, L = X.shape
    result = np.zeros((n, n), dtype=np.int32)

    row_sums = np.array(X.sum(axis=1)).ravel()  # Number of 1s in each row

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)

            dot_product = X[i:end_i] @ X[j:end_j].T
            dot_product = dot_product.todense()

            for ix in range(dot_product.shape[0]):
                for jx in range(dot_product.shape[1]):
                    result[i + ix, j + jx] = dot_product[ix, jx] + (
                        L - (row_sums[i + ix] + row_sums[j + jx] - dot_product[ix, jx])
                    )

    return result


def top_hamming_similarity_chunked(
    X: csr_matrix, chunk_size: int = 1000, k: int = 100000
) -> List[Tuple[int, int, int]]:
    """Compute pairwise hamming similarity for a sparse matrix using a chunking approach
    and return the top k similarities."""

    n, L = X.shape
    top_similarities = []  # Use a list to maintain the heap

    row_sums = np.array(X.sum(axis=1)).ravel()  # Number of 1s in each row

    for i in tqdm(range(0, n, chunk_size)):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)

            dot_product = X[i:end_i] @ X[j:end_j].T
            dot_product = dot_product.todense()

            for ix in range(dot_product.shape[0]):
                for jx in range(dot_product.shape[1]):
                    similarity = dot_product[ix, jx] + (
                        L - (row_sums[i + ix] + row_sums[j + jx] - dot_product[ix, jx])
                    )

                    # If we haven't yet accumulated k similarities, just push to the heap
                    if len(top_similarities) < k:
                        heapq.heappush(top_similarities, (similarity, i + ix, j + jx))
                    else:
                        # Compare the smallest similarity in the heap with the current similarity
                        # If the current similarity is larger, push it and pop the smallest
                        if similarity > top_similarities[0][0]:
                            heapq.heappushpop(
                                top_similarities, (similarity, i + ix, j + jx)
                            )

    # Convert heap to a sorted list of tuples (from largest similarity to smallest)
    top_similarities_sorted = sorted(top_similarities, key=lambda x: x[0], reverse=True)
    return top_similarities_sorted
