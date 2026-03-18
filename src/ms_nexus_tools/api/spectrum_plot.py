from typing import Any
from dataclasses import dataclass, field
import tomllib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotKwArgs:
    plot_kw_args: dict[str, Any] = field(default_factory=dict)
    axes_commands_and_kw_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    savefig_kw_args: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def read_config(file: str | dict, prefix: str) -> "PlotKwArgs":
        if isinstance(file, str):
            with open(file, "rb") as fle:
                file = tomllib.load(fle)

        kwargs = PlotKwArgs()
        if prefix in file:
            sub_config = file[prefix]
            if "plot" in sub_config:
                kwargs.plot_kw_args = sub_config["plot"]
            if "axes" in sub_config:
                kwargs.axes_commands_and_kw_args = sub_config["axes"]
            if "savefig" in sub_config:
                kwargs.savefig_kw_args = sub_config["savefig"]
        return kwargs


@dataclass
class ProcessArgs:
    title: str | None

    mass: np.ndarray
    spectra: np.ndarray
    target_file_name: Path

    plot_args: PlotKwArgs


def process(args: ProcessArgs) -> None:

    fig, ax = plt.subplots()
    if args.title is not None:
        fig.suptitle(args.title)

    ax.plot(args.mass, args.spectra, **args.plot_args.plot_kw_args)

    for command, kwargs in args.plot_args.axes_commands_and_kw_args.items():
        ax.__getattribute__(command)(**kwargs)

    fig.savefig(args.target_file_name, **args.plot_args.savefig_kw_args)
    plt.close(fig)
