import argparse
from dataclasses import dataclass

from ms_nexus_tools.api import args as nxargs

import pytest

from icecream import ic


@dataclass
class TestArgs(nxargs.ConfigFileArgs):
    a: int = nxargs.arg_field(doc="A test argument", required=True, default="")

    b: int = nxargs.arg_field(doc="A test argument", required=True, default="")

    c: int = nxargs.arg_field(doc="A test argument", default=-1)

    __test__ = False


@pytest.fixture(scope="session")
def config_file(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "config.toml"
    with open(fn, "w") as fle:
        fle.write("""
        [test]
        a=12

        [other]
        z = 12
        """)

    return fn


def test_normal_args():
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    args, config_dict = TestArgs.parse_args(parser, ["--a", "1", "--b", "2"])
    assert args.a == 1
    assert args.b == 2
    assert args.c == -1


def test_missing_args():
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    with pytest.raises(SystemExit):
        args, config_dict = TestArgs.parse_args(parser, ["--b", "2"])


def test_missing_args_in_config(config_file):
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    args, config_dict = TestArgs.parse_args(
        parser, ["--config", str(config_file), "--b", "2"]
    )
    assert args.a == 12
    assert args.b == 2
    assert args.c == -1


def test_missing_args_with_config(config_file):
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    with pytest.raises(SystemExit):
        args, config_dict = TestArgs.parse_args(parser, ["--config", str(config_file)])


def test_override_args_from_config(config_file):
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    args, config_dict = TestArgs.parse_args(
        parser, ["--config", str(config_file), "--a", "1", "--b", "2"]
    )
    assert args.a == 1
    assert args.b == 2
    assert args.c == -1


def test_addition_from_config(config_file):
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    args, config_dict = TestArgs.parse_args(
        parser, ["--config", str(config_file), "--a", "1", "--b", "2"]
    )
    assert config_dict == dict(other=dict(z=12))


def test_help(capsys):
    parser = argparse.ArgumentParser(prog="test")

    nxargs.add_arguments(parser, TestArgs)

    with pytest.raises(SystemExit):
        args, config_dict = TestArgs.parse_args(parser, ["--help"])

    captured = capsys.readouterr()

    assert captured.out == parser.format_help()
