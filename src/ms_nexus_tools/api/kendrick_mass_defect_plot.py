from typing import Any
from enum import Enum
from dataclasses import dataclass, field
import tomllib
from pathlib import Path
from . import compound

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotKwArgs:
    scatter_kw_args: dict[str, Any] = field(default_factory=dict)
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
            if "scatter" in sub_config:
                kwargs.scatter_kw_args = sub_config["scatter"]
            if "axes" in sub_config:
                kwargs.axes_commands_and_kw_args = sub_config["axes"]
            if "savefig" in sub_config:
                kwargs.savefig_kw_args = sub_config["savefig"]
        return kwargs


class Normalisation(Enum):
    QUADRATIC = "quad"
    LINEAR = "lin"
    LOG = "log"


@dataclass
class ProcessArgs:
    title: str | None

    mass: np.ndarray
    spectra: np.ndarray
    target_file_name: Path

    normalisation: Normalisation

    plot_args: PlotKwArgs

    normalising_formula: str = "CH2"


def process(args: ProcessArgs) -> None:

    comp_properties = compound.process(compound.ProcessArgs(args.normalising_formula))

    km = (
        args.mass
        * np.round(comp_properties.lightest_monoisotropic_mass)
        / comp_properties.lightest_monoisotropic_mass
    )
    knm = np.round(km)
    kmd = km - knm

    match args.normalisation:
        case Normalisation.LINEAR:
            sizes = args.spectra / np.max(args.spectra)
        case Normalisation.LOG:
            lg = np.log(args.spectra)
            sizes = lg / np.max(lg)
        case Normalisation.QUADRATIC:
            sizes = np.pow(args.spectra / np.max(args.spectra), 2)

    fig, ax = plt.subplots()
    if args.title is not None:
        fig.suptitle(args.title)
    ax.scatter(knm, kmd, sizes=sizes, **args.plot_args.scatter_kw_args)

    for command, kwargs in args.plot_args.axes_commands_and_kw_args.items():
        ax.__getattribute__(command)(**kwargs)

    fig.savefig(args.target_file_name, **args.plot_args.savefig_kw_args)
    plt.close(fig)
