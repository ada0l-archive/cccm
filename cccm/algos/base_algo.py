from abc import ABC

import numpy as np


class BaseAlgo(ABC):
    @classmethod
    def calc(cls, image: np.array) -> np.array:
        pass

    @classmethod
    def apply(cls, image: np.array) -> np.array:
        pass

    @classmethod
    def get_name(cls):
        pass
