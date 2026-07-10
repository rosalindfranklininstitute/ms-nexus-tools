from ms_nexus_tools.lib.dtypes import Float1D32
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SparseSampling:
    downsample_count: int = 10
    area_positions: Float1D32 = field(default_factory=lambda: np.array([15, 85, 100]))
    area_volumes: Float1D32 = field(default_factory=lambda: np.array([5, 90, 5]))

    def __post_init__(self):
        if self.area_positions.shape != self.area_volumes.shape:
            raise ValueError(
                "area_positions and area_volumes should have the same shape.",
            )
        if mx := np.max(self.area_positions) != 100:
            raise ValueError(f"area_positions should end at 100%, but found {mx:.2f}%")

    def get_edges(
        self,
        min_value: float,
        max_value: float,
        count: int,
    ) -> np.ndarray[tuple[int]]:
        count = count // self.downsample_count
        ends = np.concatenate(
            [
                [min_value],
                (max_value - min_value) * self.area_positions / 100.0 + min_value,
            ],
        )
        return np.concatenate(
            [
                *[
                    np.linspace(
                        ends[ii],
                        ends[ii + 1],
                        num=int(count * self.area_volumes[ii] / 100.0),
                        endpoint=False,
                    )
                    for ii in range(len(self.area_positions))
                ],
                [max_value],
            ],
        )
