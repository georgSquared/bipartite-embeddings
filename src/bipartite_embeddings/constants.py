from enum import auto, Enum

from utils import get_root_dir

ROOT_DIR = get_root_dir()

MOVIES_MIN_SCORE = 3.0


class SampleType(Enum):
    TRAIN = auto()
    TEST = auto()


class EdgeOperator(Enum):
    CONCAT = auto()
    COSINE = auto()
