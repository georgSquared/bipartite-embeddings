import argparse
import os
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import pandas as pd
import requests
import scipy.io as sio

from constants import ROOT_DIR, MOVIES_MIN_SCORE

# For now default to small, parametrize for other datasets
MOVIE_LENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def fetch_movie_lens():
    """
    Fetch the MovieLens dataset and store it as a csv

    Currently defaults to the small dataset
    :return:
    """
    r = requests.get(MOVIE_LENS_URL, stream=True)
    assert r.status_code == 200

    # Open the zip file and save the ratings.csv file
    with ZipFile(BytesIO(r.content)) as zipfile:
        zipfile.extract("ml-latest-small/ratings.csv", os.path.join(ROOT_DIR, "data"))


def transform_ratings_to_edgelist(min_score=3.0):
    """
    Transform the movie lens ratings.csv file to an edgelist file

    An edge is formed only if the rating is greater than a given score
    :return:
    """
    ratings_df = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "ml-latest-small", "ratings.csv")
    )
    # Filter rows below specified rating
    ratings_df = ratings_df[ratings_df["rating"] > min_score]

    # Add a prefix to user and movie ids so that they can be differentiated
    ratings_df["userId"] = "user_" + ratings_df["userId"].astype(str)
    ratings_df["movieId"] = "movie_" + ratings_df["movieId"].astype(str)

    # Clean up the dataframe
    ratings_df = ratings_df.drop_duplicates()
    ratings_df.drop(labels=["rating", "timestamp"], axis=1, inplace=True)

    ratings_df.to_csv(
        os.path.join(ROOT_DIR, "data", "ml-latest-small", "ratings_edgelist.csv"),
        index=False,
        header=False,
    )


def tranform_ppi_to_edgelist():
    mat_file = os.path.join(ROOT_DIR, "data", "ppi", "Homo_sapiens.mat")
    mat = sio.loadmat(mat_file)
    graph = nx.from_scipy_sparse_matrix(mat["network"])

    with open(os.path.join(ROOT_DIR, "data", "ppi", "edges.csv"), "wb") as edges_file:
        nx.write_edgelist(graph, edges_file, data=False, delimiter=",")


def transform_wiki_to_edgelist():
    mat_file = os.path.join(ROOT_DIR, "data", "wiki", "POS.mat")
    mat = sio.loadmat(mat_file)
    graph = nx.from_scipy_sparse_matrix(mat["network"])

    with open(os.path.join(ROOT_DIR, "data", "wiki", "edges.csv"), "wb") as edges_file:
        nx.write_edgelist(graph, edges_file, data=False, delimiter=",")


def transform_dblp_to_edgelist():
    graph = nx.read_edgelist(
        os.path.join(ROOT_DIR, "data", "dblp", "com-dblp.ungraph.txt")
    )

    # Re-write the edgelist as a csv for unified edges form
    nx.write_edgelist(
        graph,
        os.path.join(ROOT_DIR, "data", "dblp", "edges.csv"),
        delimiter=",",
        data=False,
    )


def transform_youtube_to_edgelist():
    graph = nx.read_edgelist(
        os.path.join(ROOT_DIR, "data", "youtube", "com-youtube.ungraph.txt")
    )

    # Re-write the edgelist as a csv for unified edges form
    nx.write_edgelist(
        graph,
        os.path.join(ROOT_DIR, "data", "youtube", "edges.csv"),
        delimiter=",",
        data=False,
    )


def transform_livejournal_to_edgelist():
    graph = nx.read_edgelist(
        os.path.join(ROOT_DIR, "data", "livejournal", "com-lj.ungraph.txt")
    )

    # Re-write the edgelist as a csv for unified edges form
    nx.write_edgelist(
        graph,
        os.path.join(ROOT_DIR, "data", "livejournal", "edges.csv"),
        delimiter=",",
        data=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and transform selected dataset to an edgelist"
    )

    # Add an argument to load the specified dataset, default to ml-small
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["ml-small", "blog", "ppi", "wiki", "dblp", "youtube", "livejournal"],
        required=True,
        help="Dataset to load",
    )
    parser.add_argument("-f", "--fetch", action="store_true")
    parser.add_argument("-t", "--transform", action="store_true")

    args = parser.parse_args()

    if args.dataset == "ml-small":
        if args.fetch:
            fetch_movie_lens()

        if args.transform:
            transform_ratings_to_edgelist(min_score=MOVIES_MIN_SCORE)

    elif args.dataset == "blog":
        return

    elif args.dataset == "ppi":
        tranform_ppi_to_edgelist()

    elif args.dataset == "wiki":
        transform_wiki_to_edgelist()

    elif args.dataset == "dblp":
        transform_dblp_to_edgelist()

    elif args.dataset == "youtube":
        transform_youtube_to_edgelist()

    elif args.dataset == "livejournal":
        transform_livejournal_to_edgelist()


if __name__ == "__main__":
    main()
