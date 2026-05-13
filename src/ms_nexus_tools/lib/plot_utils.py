# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple


class AxCommand(NamedTuple):
    command: str
    kwargs: dict[str, Any]
