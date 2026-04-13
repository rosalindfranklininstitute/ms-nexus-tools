from pathlib import Path

import datargs
from ..api import data_query
from ..lib.nxs_source import NxsQuerySource


def query():
    partial_args = data_query.ProcessArgs.parse_config("query")
    process_args = data_query.ProcessArgs.parse_interactive(
        "query",
        exclude=["config"],
        args=partial_args.remaining_args,
    )
    process_args.query_source = NxsQuerySource(process_args.in_path)

    data_query.process(process_args, partial_args.config)


class NxsFileTypes(datargs.FileDetails):
    def file_extension(self) -> str:
        return ".nxs"

    def filter(self, path: Path) -> bool:
        return True

    def target_name(self, in_path: Path) -> Path:
        return in_path.parent


def bulk_query():
    datargs.process_bulk(
        "query",
        data_query.ProcessArgs,
        data_query.process,
        NxsFileTypes(),
        input_arg_name="in_path",
        output_arg_name="out_dir",
    )
