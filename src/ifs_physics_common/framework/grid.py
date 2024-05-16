# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from functools import cached_property
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Optional


class DimSymbol:
    """Symbol identifying a dimension, e.g. I or I-1/2."""

    _instances: dict[int, DimSymbol] = {}

    name: str
    offset: float

    def __new__(cls, *args: Hashable) -> DimSymbol:
        key = hash(args)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, name: str, offset: float) -> None:
        self.name = name
        self.offset = offset

    def __add__(self, other: float) -> DimSymbol:
        return DimSymbol(self.name, self.offset + other)

    def __sub__(self, other: float) -> DimSymbol:
        return self + (-other)

    def __repr__(self) -> str:
        if self.offset > 0:
            return f"{self.name} + {self.offset}"
        elif self.offset < 0:
            return f"{self.name} - {-self.offset}"
        else:
            return f"{self.name}"


I = DimSymbol("I", 0)
J = DimSymbol("J", 0)
K = DimSymbol("K", 0)


class Grid:
    """Grid of points."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dims: tuple[str, ...],
        storage_shape: Optional[tuple[int, ...]] = None,
    ) -> None:
        assert len(shape) == len(dims)
        self.shape = shape
        self.dims = dims
        self.storage_shape = storage_shape or self.shape

    @cached_property
    def coords(self) -> tuple[np.ndarray, ...]:
        return tuple(np.arange(size) for size in self.storage_shape)


class ComputationalGrid:
    """A three-dimensional computational grid consisting of mass and staggered grid points."""

    grids: dict[Hashable, Grid]

    def __init__(self, nx: int, ny: int, nz: int) -> None:
        self.grids = {
            (I, J, K): Grid((nx, ny, nz), ("x", "y", "z"), (nx, ny, nz + 1)),
            (I, J, K - 1 / 2): Grid((nx, ny, nz + 1), ("x", "y", "z_h")),
            (I, J): Grid((nx, ny), ("x", "y")),
            (K,): Grid((nz,), ("z",), (nz + 1,)),
        }
