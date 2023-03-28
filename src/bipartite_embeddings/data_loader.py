import argparse
import os
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

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


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and transform MovieLens data to an edgelist"
    )

    parser.add_argument("-f", "--fetch", action="store_true")
    parser.add_argument("-t", "--transform", action="store_true")

    args = parser.parse_args()

    if args.fetch:
        fetch_movie_lens()

    if args.transform:
        transform_ratings_to_edgelist(min_score=MOVIES_MIN_SCORE)


if __name__ == "__main__":
    main()
