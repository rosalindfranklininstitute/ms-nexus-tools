# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class DataType(Enum):
    ION_H5 = "ion"
    ION_VDS = "vds"
    NEXUS = "nsx"
