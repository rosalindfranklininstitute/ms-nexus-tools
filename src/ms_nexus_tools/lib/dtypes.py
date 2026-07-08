from typing import Any
import numpy as np

Bool1D = np.ndarray[tuple[int], np.dtype[np.bool]]

Float1D32 = np.ndarray[tuple[int], np.dtype[np.float32]]
Float2D32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]

Int1D32 = np.ndarray[tuple[int], np.dtype[np.int32]]
Int2D32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]
Int3D32 = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]

Intp1D = np.ndarray[tuple[int], np.dtype[np.intp]]

Number = np.number | float | int
Number1D = np.ndarray[tuple[int], np.dtype[np.number]]
Number2D = np.ndarray[tuple[int, int], np.dtype[np.number]]

Any1D = np.ndarray[tuple[int], Any]
Any2D = np.ndarray[tuple[int, int], Any]
