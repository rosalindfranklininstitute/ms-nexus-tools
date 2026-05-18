from typing import Any
import numpy as np

Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]
Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]
Float2D32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
Int2D32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]

Number = np.number | float | int
Number1D = np.ndarray[tuple[int], np.dtype[np.number]]
Number2D = np.ndarray[tuple[int, int], np.dtype[np.number]]

Any1D = np.ndarray[tuple[int], Any]
Any2D = np.ndarray[tuple[int, int], Any]
