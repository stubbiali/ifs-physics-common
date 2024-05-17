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
from functools import cached_property, lru_cache
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numpy.typing import NDArray
    from typing import Union

    from ifs_physics_common.framework.config import DomainConfig


class AbstractDim:
    """Symbol identifying a dimension, e.g. I or I-1/2."""

    _instances: dict[int, AbstractDim] = {}

    name: str
    offset: float

    def __new__(cls, *args: Hashable) -> AbstractDim:
        key = hash(args)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, name: str, offset: float = 0) -> None:
        self.name = name
        self.offset = offset

    def __add__(self, other: float) -> AbstractDim:
        return AbstractDim(self.name, self.offset + other)

    def __sub__(self, other: float) -> AbstractDim:
        return self + (-other)

    def __eq__(self, other: Union[AbstractDim, ConcreteDim]) -> bool:
        if isinstance(other, ConcreteDim):
            return self == other.abstract_dim
        elif isinstance(other, AbstractDim):
            return self.name == other.name and self.offset == other.offset
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.name, self.offset))

    def __repr__(self) -> str:
        if self.offset > 0:
            return f"{self.name} + {self.offset}"
        elif self.offset < 0:
            return f"{self.name} - {-self.offset}"
        else:
            return f"{self.name}"

    def concretize(self, domain_config: DomainConfig) -> ConcreteDim:
        if self.name == "I":
            return ConcreteDim(self, (domain_config.xmin, domain_config.xmax), domain_config.nx)
        elif self.name == "J":
            return ConcreteDim(self, (domain_config.ymin, domain_config.ymax), domain_config.ny)
        elif self.name == "K":
            return ConcreteDim(self, (domain_config.zmin, domain_config.zmax), domain_config.nz)


I = AbstractDim("I")
J = AbstractDim("J")
K = AbstractDim("K")


class ConcreteDim:
    abstract_dim: AbstractDim
    bounds: tuple[float, float]
    size: int

    def __init__(
        self, abstract_dim: AbstractDim, bounds: tuple[float, float], domain_size: int
    ) -> None:
        self.abstract_dim = abstract_dim
        self.bounds = bounds
        self.size = domain_size + 1 if abstract_dim.offset in (-0.5, 0.5) else domain_size

    @cached_property
    def padding(self) -> int:
        return 1 if self.abstract_dim.offset in (-0.5, 0.5) else 0

    @cached_property
    def spacing(self) -> float:
        den = self.size if self.abstract_dim.offset == 0 else self.size - 1
        return abs(self.bounds[1] - self.bounds[0]) / den

    @cached_property
    def coords(self) -> NDArray:
        step = self.spacing if self.bounds[1] > self.bounds[0] else -self.spacing
        if self.abstract_dim.offset == 0:
            coords = [self.bounds[0] + 1.5 * i * step for i in range(self.size)]
        else:
            coords = [self.bounds[0] + i * step for i in range(self.size)]
        return np.array(coords)

    def __eq__(self, other: Union[AbstractDim, ConcreteDim]) -> bool:
        if isinstance(other, AbstractDim):
            return self.abstract_dim == other
        elif isinstance(other, ConcreteDim):
            return self.abstract_dim == other.abstract_dim
        else:
            return False


class Grid:
    """Grid of points."""

    abstract_dims: tuple[AbstractDim, ...]
    dims: tuple[str, ...]
    ndim: int
    padding: tuple[int, ...]
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    storage_shape: tuple[int, ...]

    def __init__(self, abstract_dims: tuple[AbstractDim, ...], domain_config: DomainConfig) -> None:
        self.abstract_dims = abstract_dims
        self.ndim = len(abstract_dims)
        self.dims = tuple(str(dim) for dim in abstract_dims)

        concrete_dims = [dim.concretize(domain_config) for dim in abstract_dims]
        self.shape = tuple(dim.size for dim in concrete_dims)
        self.coords = tuple(dim.coords for dim in concrete_dims)
        self.padding = tuple(dim.padding for dim in concrete_dims)
        self.spacing = tuple(dim.spacing for dim in concrete_dims)

    @lru_cache
    def get_storage_shape(self) -> tuple[int, ...]:
        return tuple(s + abs(p) for s, p in zip(self.shape, self.padding))

    @lru_cache
    def get_storage_origin(self) -> tuple[int, ...]:
        return tuple(max(-p, 0) for p in self.padding)

    def __repr__(self) -> str:
        out = f"{self.ndim}-D grid with dimensions:\n"
        for i in range(self.ndim):
            out += (
                f"* {self.dims[i]}: size={self.shape[i]} spacing={self.spacing[i]} "
                f"padding={self.padding[i]}"
            )
            if i < self.ndim - 1:
                out += "\n"
        return out


class ComputationalGrid:
    """A three-dimensional computational grid consisting of mass and staggered grid points."""

    GRID_LOCATIONS: tuple[tuple[AbstractDim, ...], ...] = (
        (I, J, K),
        (I, J, K - 1 / 2),
        (I, J),
        (K,),
    )

    grids: dict[Hashable, Grid]

    def __init__(self, domain_config: DomainConfig) -> None:
        self.grids = {dims: Grid(dims, domain_config) for dims in self.GRID_LOCATIONS}
