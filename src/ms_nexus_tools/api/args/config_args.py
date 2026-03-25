from typing import Any
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib

from .args import arg_field, PartialParsedArgs

from icecream import ic


@dataclass
class ConfigFileArgs:
    config: Path = arg_field(
        "-c",
        doc="The path to a configuration file. This config file can be used instead of passing files into the command line. It is a TOML formatted file. The arguments will be read out of the name of the program.",
        default=None,
        required=False,
    )

    @classmethod
    def parse_config(cls, prog: str, args=None) -> PartialParsedArgs:
        args = args if args is not None else sys.argv[1:]
        config_parser = argparse.ArgumentParser("config_parser", add_help=False)
        config_parser.add_argument("-c", "--config", default=None, type=Path)
        config_args, remaining_args = config_parser.parse_known_args(args)
        config_dict: dict[str, Any] = dict()
        config_file_args = []
        if config_args.config is not None:
            with open(config_args.config, "rb") as fle:
                config_dict = tomllib.load(fle)
            if prog in config_dict:
                for k, v in config_dict[prog].items():
                    if isinstance(v, bool):
                        config_file_args.extend([f"--{k}"])
                    elif isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, list):
                                config_file_args.extend(
                                    [f"--{k}", *[str(vvv) for vvv in vv]]
                                )
                            else:
                                config_file_args.extend([f"--{k}", str(vv)])
                    else:
                        config_file_args.extend([f"--{k}", str(v)])
                del config_dict[prog]

        return PartialParsedArgs([*config_file_args, *remaining_args], config_dict)

    @classmethod
    def parse_args(
        cls, parser: argparse.ArgumentParser, args=None
    ) -> tuple[argparse.Namespace, dict[str, Any]]:

        partial_args = ConfigFileArgs.parse_config(parser.prog, args)

        return parser.parse_args(partial_args.remaining_args), partial_args.config
