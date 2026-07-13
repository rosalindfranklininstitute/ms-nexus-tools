# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Sequence

import numpy as np

from .dtypes import Any1D, Intp1D
from .bounds import Shape


class MultiCOO(NamedTuple):
    coords: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    signal: Any1D
    axis: list[Any1D]

    def sort(self, shape) -> "MultiCOO":
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license

        linear = np.ravel_multi_index(self.coords, shape)
        if np.all(np.diff(linear) >= 0):
            return self
        order = np.argsort(linear)

        return MultiCOO(
            coords=self.coords[:, order],
            signal=self.signal[order],
            axis=[a[order] for a in self.axis],
        )

    def acc_duplicates(
        self,
        shape: Shape,
        count=False,
        signal_acc: np.ufunc = np.add,
        axis_acc: np.ufunc | Sequence[np.ufunc] = np.maximum,
    ) -> tuple["MultiCOO", Intp1D]:
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license
        linear: Intp1D = np.ravel_multi_index(self.coords, shape)
        unique_mask = np.diff(linear) != 0

        counts = np.array([], dtype=np.intp)

        if unique_mask.sum() == len(unique_mask):
            return self, np.ones((len(linear),), dtype=np.intp) if count else counts

        unique_mask = np.append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        if count:
            counts = np.diff(unique_inds)
            counts = np.append(counts, len(linear) - unique_inds[-1])

        if isinstance(axis_acc, Sequence):
            axis = [
                f.reduceat(a, unique_inds, dtype=self.signal.dtype)
                for f, a in zip(axis_acc, self.axis, strict=True)
            ]
        else:
            axis = [
                axis_acc.reduceat(a, unique_inds, dtype=self.signal.dtype)
                for a in self.axis
            ]

        return MultiCOO(
            coords=coords,
            signal=signal_acc.reduceat(
                self.signal,
                unique_inds,
                dtype=self.signal.dtype,
            ),
            axis=axis,
        ), counts
