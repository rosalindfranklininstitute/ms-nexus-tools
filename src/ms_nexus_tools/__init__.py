# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import cProfile

from . import lib as lib
from . import api as api
from .cli.ion2rfi import ion2rfi
from .cli.compound import compound
from .cli.mass_image import massimage
from .cli.spectrum_image import specimage


def main() -> None:
    print("Hello from ms-nexus-tools!")


def profile() -> None:
    cProfile.run(
        """import ms_nexus_tools; ms_nexus_tools.ion2rfi() """,
        filename="ion2rfi.perf",
    )
