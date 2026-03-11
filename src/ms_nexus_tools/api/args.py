from typing import Any
import argparse
from dataclasses import field, Field, MISSING, fields, dataclass
from enum import Enum
from pathlib import Path
import sys
import tomllib

from icecream import ic


class ArgType(Enum):
    NOT_AN_ARG = 0
    AUTOMATIC = 1
    POSITIONAL = 2
    EXPLICIT_ONLY = 3


def no_arg_field(**kw_args):
    kw_args.update(dict(metadata=dict(arg_type=ArgType.NOT_AN_ARG)))
    return field(**kw_args)


def arg_field(*args, arg_type: ArgType = ArgType.AUTOMATIC, defer=False, **kw_args):

    field_keys = (
        "default",
        "default_factory",
        "init",
        "repr",
        "hash",
        "compare",
        "metadata",
        "kw_only",
    )

    field_kw_args = {}

    for key in field_keys:
        if key in kw_args:
            field_kw_args[key] = kw_args[key]
            del kw_args[key]

    if "doc" in kw_args:
        if sys.version_info.minor >= 14:
            field_kw_args["doc"] = kw_args["doc"]
            del kw_args["doc"]
    elif "help" in kw_args:
        assert "doc" not in kw_args
        if sys.version_info.minor < 14:
            kw_args["doc"] = kw_args["help"]
        else:
            field_kw_args["doc"] = kw_args["help"]
        del kw_args["help"]

    if "metadata" in field_kw_args:
        field_kw_args.update(kw_args)
    else:
        field_kw_args["metadata"] = kw_args

    field_kw_args["metadata"]["args"] = args
    field_kw_args["metadata"]["arg_type"] = arg_type
    field_kw_args["metadata"]["defer"] = defer

    if "action" in kw_args:
        action = kw_args["action"]
        if action == "store_true":
            field_kw_args["default"] = False
        elif action == "store_false":
            field_kw_args["default"] = True

    return field(**field_kw_args)


def add_argument(parser: argparse.ArgumentParser, fld: Field):
    try:
        if "arg_type" not in fld.metadata:
            return None, None
        elif fld.metadata["arg_type"] == ArgType.NOT_AN_ARG:
            return None, None

        kw_args: dict[str, Any] = {
            "type": fld.type,
        }
        assert sys.version_info.major == 3
        if sys.version_info.minor < 14:
            kw_args["help"] = fld.metadata["doc"]
        else:
            kw_args["help"] = fld.doc

        args: list[str] = list(fld.metadata["args"])

        match fld.metadata["arg_type"]:
            case ArgType.POSITIONAL:
                assert len(args) == 0
                args.append(f"{fld.name.replace('_', '-')}")
            case ArgType.AUTOMATIC:
                args.append(f"--{fld.name.replace('_', '-')}")
            case ArgType.EXPLICIT_ONLY:
                pass

        kw_args.update(fld.metadata)

        if "doc" in kw_args:
            del kw_args["doc"]
        del kw_args["args"]
        del kw_args["arg_type"]
        del kw_args["defer"]

        if fld.metadata["arg_type"] != ArgType.POSITIONAL:
            kw_args["dest"] = fld.name

        if fld.default != MISSING:
            kw_args["default"] = fld.default

        if fld.default_factory != MISSING:
            kw_args["default"] = fld.default_factory()

        if "action" in kw_args:
            if kw_args["action"] == "store_true" or kw_args["action"] == "store_false":
                del kw_args["default"]
                del kw_args["type"]

        if "help" in kw_args and "default" in kw_args:
            kw_args["help"] = (
                str(kw_args["help"]) + f"\nDefault '{kw_args['default']}'."
            )

        if fld.metadata["defer"]:
            return None, (args, kw_args)

        return parser.add_argument(*args, **kw_args), ()
    except BaseException as e:
        raise RuntimeError(f"Could not process field '{fld.name}'") from e


def add_arguments(parser: argparse.ArgumentParser, dcls):
    defered = []
    for f in fields(dcls):
        action, args = add_argument(parser, f)
        if action is None and args is not None:
            defered.append(args)
    for args in defered:
        parser.add_argument(*args[0], **args[1])


@dataclass
class ConfigFileArgs:
    config: Path = arg_field(
        "-c",
        doc="The path to a configuration file. This config file can be used instead of passing files into the command line. It is a TOML formatted file. The arguments will be read out of the name of the program.",
        default=None,
        required=False,
    )

    @classmethod
    def parse_args(
        cls, parser: argparse.ArgumentParser, args=None
    ) -> tuple[argparse.Namespace, dict[str, Any]]:
        args = args if args is not None else sys.argv[1:]
        config_parser = argparse.ArgumentParser("config_parser", add_help=False)
        config_parser.add_argument("-c", "--config", default=None, type=Path)
        config_args, remaining_args = config_parser.parse_known_args(args)
        config_dict = {}
        config_file_args = []
        if config_args.config is not None:
            with open(config_args.config, "rb") as fle:
                config_dict = tomllib.load(fle)
            for k, v in config_dict[parser.prog].items():
                config_file_args.extend([f"--{k}", str(v)])
            del config_dict[parser.prog]

        return parser.parse_args([*config_file_args, *remaining_args]), config_dict
