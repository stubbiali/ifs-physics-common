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
from contextlib import contextmanager
import numpy as np
from typing import TYPE_CHECKING

import gt4py
from sympl._core.data_array import DataArray

from ifs_physics_common import numpyx as npx

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal, Optional

    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import (
        AbstractGridDimTuple,
        ComputationalGrid,
        DataDimTuple,
        DimTuple,
    )
    from ifs_physics_common.utils.typingx import NDArrayLike


def gt_zeros(
    computational_grid: ComputationalGrid,
    grid_dims: AbstractGridDimTuple,
    data_dims: Optional[DataDimTuple] = None,
    *,
    gt4py_config: GT4PyConfig,
    dtype_name: Literal["bool", "float", "int"],
) -> NDArrayLike:
    """
    Create an array defined over the grid ``grid_id`` of ``computational_grid``
    and fill it with zeros.

    Relying on GT4Py utilities to optimally allocate memory based on the chosen backend.
    """
    grid = computational_grid.grids[grid_dims]
    data_dims = data_dims or ()
    data_shape = tuple(dim.size for dim in data_dims)
    shape = grid.get_storage_shape() + data_shape
    dtype = gt4py_config.dtypes.from_name(dtype_name)
    return gt4py.storage.zeros(shape, dtype, backend=gt4py_config.backend)


def get_data_array(
    buffer: NDArrayLike,
    computational_grid: ComputationalGrid,
    grid_dims: AbstractGridDimTuple,
    units: str,
    data_dims: Optional[DataDimTuple] = None,
) -> DataArray:
    """Create a ``DataArray`` out of  a raw ``buffer``."""
    grid = computational_grid.grids[grid_dims]
    data_dims = data_dims or ()
    dim_names = grid.dim_names + tuple(str(dim) for dim in data_dims)
    coords = grid.coords + tuple(dim.coords for dim in data_dims)
    view_shape = grid.shape + tuple(dim.size for dim in data_dims)
    view_slice = grid.get_storage_view_slice() + tuple(slice(0, dim.size) for dim in data_dims)
    return DataArray(
        buffer,
        dims=dim_names,
        coords=coords,
        attrs={"units": units, "view_shape": view_shape, "view_slice": view_slice},
    )


def zeros(
    computational_grid: ComputationalGrid,
    grid_dims: AbstractGridDimTuple,
    units: str,
    data_dims: Optional[DataDimTuple] = None,
    *,
    gt4py_config: GT4PyConfig,
    dtype_name: Literal["bool", "float", "int"],
) -> DataArray:
    """
    Create a ``DataArray`` defined over the grid ``grid_id`` of ``computational_grid``
    and fill it with zeros.
    """
    buffer = gt_zeros(
        computational_grid,
        grid_dims,
        data_dims=data_dims,
        gt4py_config=gt4py_config,
        dtype_name=dtype_name,
    )
    return get_data_array(buffer, computational_grid, grid_dims, units, data_dims=data_dims)


def assign(field: DataArray, buffer: NDArrayLike) -> None:
    assert field.ndim == buffer.ndim
    view_shape = field.attrs.get("view_shape", field.shape)
    view_slice = field.attrs.get("view_slice", tuple(slice(0, None) for _ in range(field.ndim)))
    buffer_slice = tuple(slice(0, view_size) for view_size in view_shape)
    reps = tuple(
        (view_size - 1) // buffer_size + 1
        for view_size, buffer_size in zip(view_shape, buffer.shape)
    )
    buffer = np.tile(buffer, reps)
    npx.assign(field.data[view_slice], buffer[buffer_slice])


TEMPORARY_STORAGE_POOL: dict[int, list[NDArrayLike]] = {}


@contextmanager
def managed_temporary_storage(
    computational_grid: ComputationalGrid,
    *args: tuple[DimTuple, Literal["bool", "float", "int"]],
    gt4py_config: GT4PyConfig,
) -> Iterator[NDArrayLike]:
    """
    Get temporary storages defined over the grids of ``computational_grid``.

    Each ``arg`` is a tuple where the first element specifies the grid identifier, and the second
    element specifies the datatype.

    The storages are either created on-the-fly, or retrieved from ``TEMPORARY_STORAGE_POOL``
    if available. On exit, all storages are included in ``TEMPORARY_STORAGE_POOL`` for later use.
    """
    grid_hashes = []
    storages = []
    for grid_dims, dtype_name in args:
        grid = computational_grid.grids[grid_dims]
        grid_hash = hash((grid.shape + grid_dims, dtype_name))
        pool = TEMPORARY_STORAGE_POOL.setdefault(grid_hash, [])
        if len(pool) > 0:
            storage = pool.pop()
        else:
            storage = gt_zeros(
                computational_grid, grid_dims, gt4py_config=gt4py_config, dtype_name=dtype_name
            )
        grid_hashes.append(grid_hash)
        storages.append(storage)

    try:
        if len(storages) == 1:
            yield storages[0]
        else:
            yield storages
    finally:
        for grid_hash, storage in zip(grid_hashes, storages):
            TEMPORARY_STORAGE_POOL[grid_hash].append(storage)


@contextmanager
def managed_temporary_storage_pool() -> Iterator[None]:
    """
    Clear the pool of temporary storages ``TEMPORARY_STORAGE_POOL`` on entry and exit.

    Useful when running multiple simulations using different backends within the same session.
    All simulations using the same backend should be wrapped by this context manager.
    """
    try:
        TEMPORARY_STORAGE_POOL.clear()
        yield None
    finally:
        for grid_hash, storages in TEMPORARY_STORAGE_POOL.items():
            num_storages = len(storages)
            for _ in range(num_storages):
                storage = storages.pop()
                del storage
        TEMPORARY_STORAGE_POOL.clear()


# if __name__ == "__main__":
#     from ifs_physics_common.framework.config import DomainConfig, GT4PyConfig
#     from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
#     from ifs_physics_common.utils.numpyx import assign as npx_assign
#
#     domain_config = DomainConfig(nx=1, ny=14)
#     gt4py_config = GT4PyConfig(backend="numpy")
#     grid = ComputationalGrid(domain_config)
#     field_ = zeros(grid, (I, J, K), units="K", gt4py_config=gt4py_config, dtype_name="float")
#     np.random.seed(20220604)
#     buffer_ = np.random.rand(5, 5, 1)
#     initialize_field(field_, buffer_)
#     print(f"{buffer_[..., 0]=}")
#     print(f"{field_.data[..., 0]=}")
