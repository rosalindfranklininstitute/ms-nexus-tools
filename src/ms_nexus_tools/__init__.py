# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from . import lib as lib
from . import api as api
from .cli.compound import compound
from .cli.inspect_h5 import inspect
from .cli.query import query, bulk_query


def main() -> None:
    print("Hello from ms-nexus-tools!")
