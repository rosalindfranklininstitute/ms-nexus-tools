# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from . import lib as lib
from . import api as api
from .cli.ion2rfi import ion2rfi


def main() -> None:
    print("Hello from ms-nexus-tools!")
