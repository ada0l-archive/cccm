import numpy as np
import cv2 as cv
from .base_algo import BaseAlgo


class GrayWorld(BaseAlgo):
    @classmethod
    def calc(cls, image: np.array) -> np.array:
        b, g, r = cv.split(image)
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)

        k_b = (avg_g + avg_r) / (2 * avg_b)
        k_g = 1.0
        k_r = (avg_g + avg_b) / (2 * avg_r)

        correction_matrix = np.array([[k_b, 0, 0], [0, k_g, 0], [0, 0, k_r]])
        return correction_matrix

    @classmethod
    def apply(cls, image: np.array) -> np.array:
        return cv.transform(image, cls.calc(image))

    @classmethod
    def get_name(cls):
        return "gray_world"
