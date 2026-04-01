from enum import Enum
import numpy as np


class Norm(Enum):
    NONE = "none"
    MAX = "max"
    TIC = "tic"


EMPTY = np.full((1,), np.nan)


def normalise(data, norm: Norm):
    match norm:
        case Norm.NONE:
            return data
        case Norm.TIC:
            return data / np.sum(data)
        case Norm.MAX:
            return data / np.max(data)


def norm(data, norm: Norm):
    match norm:
        case Norm.NONE:
            return 1
        case Norm.TIC:
            return np.sum(data)
        case Norm.MAX:
            return np.max(data)


def _should_operate(ndims, axis):
    if axis is None:
        return True
    elif isinstance(axis, int):
        return axis < ndims
    elif len(axis) != 1:
        return True
    else:
        return axis[0] < ndims


def _operate(operation, current, new, axis):
    if _should_operate(len(new.shape), axis):
        inc_value = operation(new, axis=axis)
    else:
        inc_value = new
    if np.isnan(current).all():
        return inc_value
    else:
        return operation([current, inc_value], axis=0)


class IncrementalNorm:
    def __init__(self, axis=None):
        self.max: np.ndarray = EMPTY
        self.tic: np.ndarray = EMPTY
        self.axis: tuple[int, ...] | None = axis

    def add(self, data, axis=None):

        axis_to_use = axis if axis is not None else self.axis
        self.max = _operate(np.nanmax, self.max, data, axis_to_use)
        self.tic = _operate(np.nansum, self.tic, data, axis_to_use)

    def __getitem__(self, index: str | Norm):
        match index:
            case Norm.MAX | Norm.MAX.value:
                return self.max
            case Norm.TIC | Norm.TIC.value:
                return self.tic
