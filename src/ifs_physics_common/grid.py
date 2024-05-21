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
from collections import namedtuple
from functools import cached_property, lru_cache
import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numpy.typing import NDArray
    from typing import Literal, Optional

    from ifs_physics_common.config import GridConfig


DIM_INSTANCES = {}


class MetaDim(type):
    def __call__(cls, *args):
        key = hash(args)
        if key not in DIM_INSTANCES:
            obj = super().__call__(*args)
            DIM_INSTANCES[key] = obj
        return DIM_INSTANCES[key]


class AbstractGridDim(metaclass=MetaDim):
    name: str
    axis: Literal[0, 1, 2]
    offset: float
    direction: Literal[-1, 1]

    def __init__(
        self, name: str, axis: Literal[0, 1, 2], offset: float = 0, direction: Literal[-1, 1] = 1
    ) -> None:
        assert axis in (0, 1, 2)
        assert offset in (-0.5, 0, 0.5)
        self.name = name
        self.axis = axis
        self.offset = offset
        self.direction = direction

    def __add__(self, other: float) -> AbstractGridDim:
        return AbstractGridDim(self.name, self.axis, self.offset + other)

    def __sub__(self, other: float) -> AbstractGridDim:
        return self + (-other)

    def __neg__(self) -> AbstractGridDim:
        return AbstractGridDim(self.name, self.axis, self.offset, -self.direction)

    def __eq__(self, other: Union[AbstractGridDim, ConcreteGridDim]) -> bool:
        if isinstance(other, ConcreteGridDim):
            return self == other.abstract_dim
        elif isinstance(other, AbstractGridDim):
            return (
                self.name == other.name
                and self.axis == other.axis
                and self.offset == other.offset
                and self.direction == other.direction
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.name, self.axis, self.offset, self.direction))

    def __repr__(self) -> str:
        if self.offset > 0:
            return f"{self.name} + {self.offset}"
        elif self.offset < 0:
            return f"{self.name} - {-self.offset}"
        else:
            return f"{self.name}"

    def concretize(self, grid_config: GridConfig) -> ConcreteGridDim:
        if self.axis == 0:
            return ConcreteGridDim(self, (grid_config.xmin, grid_config.xmax), grid_config.nx)
        elif self.axis == 1:
            return ConcreteGridDim(self, (grid_config.ymin, grid_config.ymax), grid_config.ny)
        elif self.axis == 2:
            return ConcreteGridDim(self, (grid_config.zmin, grid_config.zmax), grid_config.nz)
        else:
            raise ValueError(f"Unknown dimension {repr(self.name)}.")


I = AbstractGridDim("I", 0)
IJ = AbstractGridDim("IJ", 0)
J = AbstractGridDim("J", 1)
K = AbstractGridDim("K", 2)


class ConcreteGridDim:
    abstract_dim: AbstractGridDim
    bounds: tuple[float, float]
    size: int

    def __init__(
        self, abstract_dim: AbstractGridDim, bounds: tuple[float, float], domain_size: int
    ) -> None:
        self.abstract_dim = abstract_dim
        self.bounds = bounds
        self.size = domain_size + 1 if abstract_dim.offset in (-0.5, 0.5) else domain_size

    @cached_property
    def padding(self) -> int:
        return self.abstract_dim.direction if self.abstract_dim.offset == 0 else 0

    @cached_property
    def spacing(self) -> float:
        den = self.size if self.abstract_dim.offset == 0 else self.size - 1
        return abs(self.bounds[1] - self.bounds[0]) / den

    @cached_property
    def coords(self) -> NDArray:
        step = self.spacing if self.bounds[1] > self.bounds[0] else -self.spacing
        step_factor = 1.5 if self.abstract_dim.offset == 0 else 1.0
        coords = [
            self.bounds[0] + i * step_factor * step for i in range(self.size + abs(self.padding))
        ]
        if self.abstract_dim.direction < 0:
            coords = coords[::-1]
        return np.array(coords)

    def __eq__(self, other: Union[AbstractGridDim, ConcreteGridDim]) -> bool:
        if isinstance(other, AbstractGridDim):
            return self.abstract_dim == other
        elif isinstance(other, ConcreteGridDim):
            return self.abstract_dim == other.abstract_dim
        else:
            return False


ExpandedDim = namedtuple("ExpandedDim", field_names=[])


class DataDim(metaclass=MetaDim):
    name: str
    size: int
    direction: Literal[-1, 1]
    index: Optional[int]

    def __init__(
        self, name: str, size: int, direction: Literal[-1, 1] = 1, index: Optional[int] = None
    ) -> None:
        self.name = name
        assert size > 0
        self.size = size
        self.direction = direction
        self.index = index

    def __neg__(self) -> DataDim:
        return DataDim(self.name, self.size, -self.direction, self.index)

    def __eq__(self, other: DataDim) -> bool:
        if isinstance(other, DataDim):
            return (
                self.name == other.name
                and self.size == other.size
                and self.direction == other.direction
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.name, self.size, self.direction))

    def __getitem__(self, index: int) -> DataDim:
        assert self.index is None
        return DataDim(self.name, self.size, self.direction, index)

    def __repr__(self) -> str:
        return f"{self.name}{self.size}"

    @cached_property
    def coords(self) -> NDArray:
        return np.arange(start=0, stop=self.size, step=self.direction)


D5 = DataDim("D", 5)


AbstractGridDimTuple = tuple[AbstractGridDim, ...]
DataDimTuple = tuple[DataDim, ...]
DimTuple = tuple[Union[AbstractGridDim, DataDim, type[ExpandedDim]], ...]


class Grid:
    """Grid of points."""

    abstract_dims: AbstractGridDimTuple
    dim_names: tuple[str, ...]
    ndim: int
    padding: tuple[int, ...]
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    storage_shape: tuple[int, ...]

    def __init__(self, abstract_dims: AbstractGridDimTuple, grid_config: GridConfig) -> None:
        self.abstract_dims = abstract_dims
        self.ndim = len(abstract_dims)
        self.dim_names = tuple(str(dim) for dim in abstract_dims)

        concrete_dims = [dim.concretize(grid_config) for dim in abstract_dims]
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

    @lru_cache
    def get_storage_view_slice(self) -> tuple[slice, ...]:
        origin = self.get_storage_origin()
        return tuple(slice(o, o + s) for o, s in zip(origin, self.shape))

    def __repr__(self) -> str:
        out = f"{self.ndim}-D grid with dimensions: \n"
        for i in range(self.ndim):
            out += (
                f"* {self.dim_names[i]}: size={self.shape[i]} spacing={self.spacing[i]} "
                f"padding={self.padding[i]}"
            )
            if i < self.ndim - 1:
                out += "\n"
        return out


class ComputationalGrid:
    """A three-dimensional computational grid consisting of mass and staggered grid points."""

    GRID_LOCATIONS: tuple[AbstractGridDimTuple, ...] = (
        (I, J, K),
        (I, J, K - 1 / 2),
        (I, J),
        (K,),
        (K - 1 / 2,),
    )

    grid_config: GridConfig
    grids: dict[Hashable, Grid]

    def __init__(self, grid_config: GridConfig) -> None:
        self.grid_config = grid_config
        self.grids = {dims: Grid(dims, grid_config) for dims in self.GRID_LOCATIONS}
