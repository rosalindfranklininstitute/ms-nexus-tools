from typing import Any
from dataclasses import dataclass, field
import tomllib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotKwArgs:
    imshow_kw_args: dict[str, Any] = field(default_factory=dict)
    axes_commands_and_kw_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    colorbar_kw_args: dict[str, Any] = field(default_factory=dict)
    savefig_kw_args: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def read_config(file: str | dict, prefix: str) -> "PlotKwArgs":
        if isinstance(file, str):
            with open(file, "rb") as fle:
                file = tomllib.load(fle)

        kwargs = PlotKwArgs()
        if prefix in file:
            sub_config = file[prefix]
            if "imshow" in sub_config:
                kwargs.imshow_kw_args = sub_config["imshow"]
            if "axes" in sub_config:
                kwargs.axes_commands_and_kw_args = sub_config["axes"]
            if "colorbar" in sub_config:
                kwargs.colorbar_kw_args = sub_config["colorbar"]
            if "savefig" in sub_config:
                kwargs.savefig_kw_args = sub_config["savefig"]
        return kwargs


@dataclass
class ProcessArgs:
    total_ion_count: np.ndarray
    target_file_name: Path

    plot_args: PlotKwArgs


def process(args: ProcessArgs) -> None:

    fig, ax = plt.subplots()
    im = ax.imshow(args.total_ion_count, **args.plot_args.imshow_kw_args)

    for command, kwargs in args.plot_args.axes_commands_and_kw_args.items():
        ax.__getattribute__(command)(**kwargs)

    fig.colorbar(im, ax=ax, **args.plot_args.colorbar_kw_args)

    fig.savefig(args.target_file_name, **args.plot_args.savefig_kw_args)
