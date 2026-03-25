import copy
from typing import Any, get_args, NamedTuple, Optional, Literal
import argparse
from dataclasses import field, Field, MISSING, fields
from enum import Enum
import sys

from icecream import ic

MISSING_TYPE = type(MISSING)

PossibleActions = Literal["store", "store_true", "store_false", "append"]


class ArgType(Enum):
    NOT_AN_ARG = 0
    AUTOMATIC = 1
    POSITIONAL = 2
    EXPLICIT_ONLY = 3


class PartialParsedArgs(NamedTuple):
    remaining_args: list[str]
    config: dict[str, Any]


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

    if "metadata" not in field_kw_args:
        field_kw_args["metadata"] = dict()

    if "doc" in kw_args:
        assert "help" not in kw_args
        if sys.version_info.minor < 14:
            field_kw_args["metadata"]["doc"] = kw_args["doc"]
            del kw_args["doc"]
    elif "help" in kw_args:
        assert "doc" not in kw_args
        if sys.version_info.minor < 14:
            field_kw_args["metadata"]["doc"] = kw_args["help"]
        else:
            kw_args["doc"] = kw_args["help"]
        del kw_args["help"]
    else:
        if sys.version_info.minor < 14:
            field_kw_args["metadata"]["doc"] = ""
        else:
            kw_args["doc"] = ""

    #     field_kw_args.update(kw_args)
    # else:
    #     field_kw_args["metadata"] = kw_args

    field_kw_args["metadata"]["args"] = args
    field_kw_args["metadata"]["arg_type"] = arg_type
    field_kw_args["metadata"]["defer"] = defer
    field_kw_args["metadata"]["kw_args"] = kw_args

    if "action" in kw_args:
        action = kw_args["action"]
        if action == "store_true":
            field_kw_args["default"] = False
        elif action == "store_false":
            field_kw_args["default"] = True

    return field(**field_kw_args)


class Action:
    aliases: list[str]

    arg_type: ArgType
    defer: bool

    action: PossibleActions
    value_type: Any | MISSING_TYPE
    help: str | MISSING_TYPE
    default: Any | MISSING_TYPE
    nargs: None | str | int
    choices: list[Any] | None
    required: bool
    dest: str
    metavar: str | tuple[str] | None
    extra_kw_args: dict[str, Any]

    def __getitem__(self, index):
        match index:
            case "aliases":
                return self.aliases
            case "arg_type":
                return self.arg_type
            case "defer":
                return self.defer
            case "action":
                return self.action
            case "value_type":
                return self.value_type
            case "help":
                return self.help
            case "default":
                return self.default
            case "nargs":
                return self.nargs
            case "choices":
                return self.choices
            case "required":
                return self.required
            case "dest":
                return self.dest
            case "metavar":
                return self.metavar
            case "extra_kw_args":
                return self.extra_kw_args
            case _:
                raise ValueError(f"Unknown value for [{index}]")

    def fields(self) -> tuple[str, ...]:
        return (
            "aliases",
            "arg_type",
            "defer",
            "action",
            "value_type",
            "help",
            "default",
            "nargs",
            "choices",
            "required",
            "dest",
            "metavar",
            "extra_kw_args",
        )

    def __repr__(self) -> str:
        parts = []
        for k in self.fields():
            parts.append(f"{k}={self[k]}")
        return f"Action( {', '.join(parts)})"

    def to_argument_kwargs(self) -> dict[str, Any]:
        result: dict[str, Any] = dict(action=self.action, required=self.required)

        def add_if_not(key, filter_value):
            if self[key] != filter_value:
                result[key] = self[key]

        if self.value_type != MISSING:
            result["type"] = self.value_type

        if self.arg_type != ArgType.POSITIONAL:
            result["dest"] = self.dest
        add_if_not("help", MISSING)
        add_if_not("default", MISSING)
        add_if_not("nargs", None)
        add_if_not("choices", None)
        add_if_not("metavar", None)
        for k, v in self.extra_kw_args:
            result[k] = v
        return result

    def get_display_name(self) -> str:
        return self.aliases[-1].strip("-").replace("-", " ")


def parse_field(fld: Field) -> Optional["Action"]:
    try:
        if "arg_type" not in fld.metadata:
            return None
        elif fld.metadata["arg_type"] == ArgType.NOT_AN_ARG:
            return None

        action = Action()

        # default
        # default_factory
        if fld.default != MISSING:
            assert fld.default_factory == MISSING
            action.default = fld.default
        elif fld.default_factory != MISSING:
            assert fld.default == MISSING
            action.default = fld.default_factory()
        else:
            action.default = MISSING

        # init
        # repr
        # hash
        # compare
        # kw_only
        # -> ignore
        # doc
        assert sys.version_info.major == 3
        if sys.version_info.minor < 14:
            action.help = fld.metadata["doc"]
        else:
            action.help = fld.doc

        if action.default != MISSING:
            action.help += f"Default '{action.default}'."

        # metadata
        action.aliases = list(fld.metadata["args"])
        action.arg_type = fld.metadata["arg_type"]
        action.defer = fld.metadata["defer"]

        match action.arg_type:
            case ArgType.POSITIONAL:
                assert len(action.aliases) == 0
                action.aliases.append(f"{fld.name.replace('_', '-')}")
            case ArgType.AUTOMATIC:
                action.aliases.append(f"--{fld.name.replace('_', '-')}")
            case ArgType.EXPLICIT_ONLY:
                pass
        action.dest = fld.name

        kw_args = copy.copy(fld.metadata["kw_args"])
        action.action = kw_args.get("action", "store")

        match action.action:
            case "store_true" | "store_false":
                action.default = MISSING
                action.value_type = MISSING
                assert "nargs" not in kw_args
            case "append":
                inner_type = get_args(fld.type)
                assert len(inner_type) == 1
                action.value_type = inner_type[0]
            case _:
                action.value_type = fld.type

        action.nargs = kw_args.get("nargs", None)
        if isinstance(action.nargs, int):
            nargs_more_than_one = action.nargs > 1
        elif isinstance(action.nargs, str):
            nargs_more_than_one = action.nargs == "*" or action.nargs == "+"
        else:
            nargs_more_than_one = False

        if nargs_more_than_one:
            inner_type = get_args(action.value_type)
            assert len(inner_type) == 1
            action.value_type = inner_type[0]

        action.choices = kw_args.get("choices", None)
        action.required = kw_args.get("required", False)
        action.metavar = kw_args.get("metavar", None)

        for name in action.fields():
            if name in kw_args:
                del kw_args[name]
        action.extra_kw_args = kw_args

        return action
    except BaseException as e:
        raise RuntimeError(f"Could not process field '{fld.name}'") from e


def parse_fields(dcls) -> list[Action]:
    return [f for f in [parse_field(f) for f in fields(dcls)] if f is not None]


def add_argument(parser: argparse.ArgumentParser, fld: Field | Action):
    if isinstance(fld, Action):
        action = fld
    else:
        action = parse_field(fld)

    if action is None:
        return None, None
    return parser.add_argument(*action.aliases, **action.to_argument_kwargs()), None


def add_arguments(parser: argparse.ArgumentParser, dcls: type | list[Action]):
    print()
    print("---")
    ic()
    print("---")
    if isinstance(dcls, type):
        actions = parse_fields(dcls)
    else:
        actions = dcls
    defered = []
    for a in actions:
        if a.defer:
            defered.append(a)
        else:
            add_argument(parser, a)
    for a in defered:
        add_argument(parser, a)
    print("<<<")
