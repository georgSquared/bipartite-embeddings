from networkx.algorithms import bipartite

from bipartite_embeddings.utils import load_graph


def test_load_movie_lens_small():
    """
    Test loading the movie lens small dataset
    :return:
    """
    G = load_graph()

    assert bipartite.is_bipartite(G)


if __name__ == "__main__":
    test_load_movie_lens_small()
