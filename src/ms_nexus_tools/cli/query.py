import argparse

import datargs
from ..api import data_query


def query():
    partial_args = data_query.ProcessArgs.parse_config("query")
    process_args = data_query.ProcessArgs.parse_interactive(
        "query",
        exclude=["config"],
        args=partial_args.remaining_args,
    )
    data_query.process(process_args, partial_args.config)
