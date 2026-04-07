from enum import Enum
import numpy as np


class OriginLocation(Enum):
    UPPER_LEFT = "upper left"
    UPPER_RIGHT = "upper right"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"


def adjust_origin(
    a: np.ndarray,
    new: OriginLocation,
    current: OriginLocation = OriginLocation.UPPER_LEFT,
):

    left = (OriginLocation.UPPER_LEFT, OriginLocation.LOWER_LEFT)
    right = (OriginLocation.UPPER_RIGHT, OriginLocation.LOWER_RIGHT)

    upper = (OriginLocation.UPPER_LEFT, OriginLocation.UPPER_RIGHT)
    lower = (OriginLocation.LOWER_LEFT, OriginLocation.LOWER_RIGHT)

    axis = []
    if (current in left and new in right) or (current in right and new in left):
        axis.append(1)
    if (current in upper and new in lower) or (current in lower and new in upper):
        axis.append(0)

    if len(axis) > 0:
        return np.flip(a, axis=tuple(axis))
    else:
        return a
