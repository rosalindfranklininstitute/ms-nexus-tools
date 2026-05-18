# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar

from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class Plottable(Generic[T]):
    title: str
    color: str
    value: T
