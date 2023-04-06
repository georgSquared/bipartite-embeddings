from enum import StrEnum, auto

from utils import get_root_dir

ROOT_DIR = get_root_dir()

MOVIES_MIN_SCORE = 3.0


class SampleType(StrEnum):
    TRAIN = auto()
    TEST = auto()


class EdgeOperator(StrEnum):
    CONCAT = auto()
    COSINE = auto()
