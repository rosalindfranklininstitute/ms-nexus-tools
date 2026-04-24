from typing import Any
from dataclasses import dataclass, field
import tomllib
from pathlib import Path
from bisect import bisect_left, bisect_right

import numpy as np
import matplotlib.pyplot as plt

from ..lib.plot_utils import AxCommand

from . import compound
from . import image_plot
from . import spectrum_plot


@dataclass
class PlotKwArgs:
    plot_kw_args: dict[str, Any] = field(default_factory=dict)
    plot_axes_commands_and_kw_args: list[AxCommand] = field(default_factory=list)

    imshow_kw_args: dict[str, Any] = field(default_factory=dict)
    imshow_axes_commands_and_kw_args: list[AxCommand] = field(default_factory=list)
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
            if "plot" in sub_config:
                kwargs.plot_kw_args = sub_config["plot"]
            if "plot_axes" in sub_config:
                kwargs.plot_axes_commands_and_kw_args = sub_config["plot_axes"]
            if "imshow" in sub_config:
                kwargs.imshow_kw_args = sub_config["imshow"]
            if "imshow_axes" in sub_config:
                kwargs.imshow_axes_commands_and_kw_args = sub_config["imshow_axes"]
            if "savefig" in sub_config:
                kwargs.savefig_kw_args = sub_config["savefig"]
        return kwargs

    @staticmethod
    def from_image_and_spectra_args(
        image_plot_args: image_plot.PlotKwArgs,
        spectrum_plot_args: spectrum_plot.PlotKwArgs,
    ):
        return PlotKwArgs(
            plot_kw_args=spectrum_plot_args.plot_kw_args,
            plot_axes_commands_and_kw_args=spectrum_plot_args.axes_commands_and_kw_args,
            imshow_kw_args=image_plot_args.imshow_kw_args,
            imshow_axes_commands_and_kw_args=image_plot_args.axes_commands_and_kw_args,
            colorbar_kw_args=image_plot_args.colorbar_kw_args,
            savefig_kw_args=spectrum_plot_args.savefig_kw_args
            | image_plot_args.savefig_kw_args,
        )


@dataclass
class ProcessArgs:
    title: str | None

    mass: np.ndarray
    spectra: np.ndarray
    image: np.ndarray
    target_file_name: Path

    plot_args: PlotKwArgs


def process(args: ProcessArgs) -> None:

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    if args.title is not None:
        fig.suptitle(args.title)

    axs[1].set_title("Mass spectrum")
    axs[1].plot(args.mass, args.spectra, **args.plot_args.plot_kw_args)
    axs[1].set_xlabel("m/z")

    for command in args.plot_args.plot_axes_commands_and_kw_args:
        axs[1].__getattribute__(command.command)(**command.kwargs)

    axs[0].set_title("Mass image")
    im = axs[0].imshow(args.image, **args.plot_args.imshow_kw_args)

    for command in args.plot_args.imshow_axes_commands_and_kw_args:
        axs[0].__getattribute__(command.command)(**command.kwargs)

    im_min = np.min(args.image)
    im_max = np.max(args.image)
    fig.colorbar(
        im,
        ax=axs[0],
        location="right",
        shrink=0.8,
        ticks=np.linspace(im_min, im_max, 6),
        **args.plot_args.colorbar_kw_args,
    )

    fig.savefig(args.target_file_name, **args.plot_args.savefig_kw_args)
    plt.close(fig)
