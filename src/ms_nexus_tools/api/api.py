from typing import Any
import argparse
from dataclasses import field, Field, MISSING, fields
from enum import Enum


class ArgType(Enum):
    AUTOMATIC = 1
    POSITIONAL = 2
    EXPLICIT_ONLY = 3


def arg_field(*args, arg_type: ArgType = ArgType.AUTOMATIC, **kw_args):

    field_keys = (
        "default",
        "default_factory",
        "init",
        "repr",
        "hash",
        "compare",
        "metadata",
        "kw_only",
        "doc",
    )

    field_kw_args = {}

    for key in field_keys:
        if key in kw_args:
            field_kw_args[key] = kw_args[key]
            del kw_args[key]

    if "help" in kw_args:
        field_kw_args["doc"] = kw_args["help"]
        del kw_args["help"]

    if "metadata" in field_kw_args:
        field_kw_args.update(kw_args)
    else:
        field_kw_args["metadata"] = kw_args

    field_kw_args["metadata"]["args"] = args
    field_kw_args["metadata"]["arg_type"] = arg_type

    if "action" in kw_args:
        action = kw_args["action"]
        if action == "store_true":
            field_kw_args["default"] = False
        elif action == "store_false":
            field_kw_args["default"] = True

    return field(**field_kw_args)


def add_argument(parser: argparse.ArgumentParser, fld: Field):
    try:
        kw_args: dict[str, Any] = {
            "type": fld.type,
            "help": fld.doc,
        }
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
        del kw_args["args"]
        del kw_args["arg_type"]
        kw_args["dest"] = fld.name

        if fld.default != MISSING:
            kw_args["default"] = fld.default

        if "action" in kw_args:
            if kw_args["action"] == "store_true" or kw_args["action"] == "store_false":
                del kw_args["default"]
                del kw_args["type"]

        if "help" in kw_args and "default" in kw_args:
            kw_args["help"] = (
                str(kw_args["help"]) + f"\nDefault '{kw_args['default']}'."
            )

        return parser.add_argument(*args, **kw_args)
    except BaseException as e:
        raise RuntimeError(f"Could not process field '{fld.name}'") from e


def add_arguments(parser: argparse.ArgumentParser, dcls):
    for f in fields(dcls):
        add_argument(parser, f)
