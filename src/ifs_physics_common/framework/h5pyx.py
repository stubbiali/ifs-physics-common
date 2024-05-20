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
from h5py import File
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import DataDim, ExpandedDim
from ifs_physics_common.framework.storage import assign, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Literal, Optional

    from sympl._core.typingx import DataArray

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import (
        AbstractGridDimTuple,
        ComputationalGrid,
        DataDimTuple,
        DimTuple,
    )


class HDF5Operator:
    computational_grid: ComputationalGrid
    f: File
    gt4py_config: GT4PyConfig

    def __init__(
        self, computational_grid: ComputationalGrid, filename: str, *, gt4py_config: GT4PyConfig
    ) -> None:
        self.computational_grid = computational_grid
        self.f = File(filename, mode="r")
        self.gt4py_config = gt4py_config

    def get_field(
        self,
        grid_dims: AbstractGridDimTuple,
        units: str,
        data_dims: Optional[DataDimTuple] = None,
        *,
        dtype_name: Literal["bool", "float", "int"],
        h5_name: str,
        h5_dims: DimTuple,
        h5_dims_map: DimTuple,
    ) -> DataArray:
        data_dims = data_dims or ()
        field = zeros(
            self.computational_grid,
            grid_dims,
            units,
            data_dims=data_dims,
            gt4py_config=self.gt4py_config,
            dtype_name=dtype_name,
        )
        rhs = self.read_field(grid_dims + data_dims, dtype_name, h5_name, h5_dims, h5_dims_map)
        assign(field, rhs)
        return field

    def read_field(
        self,
        dims: DimTuple,
        dtype_name: Literal["bool", "float", "int"],
        h5_name: str,
        h5_dims: DimTuple,
        h5_dims_map: DimTuple,
    ) -> NDArray:
        ds = self.f.get(h5_name, None)
        if ds is None:
            raise RuntimeError(f"Field {repr(h5_name)} not found in HDF5 file.")
        value = np.asarray(ds[...])

        if value.ndim != len(h5_dims):
            raise RuntimeError(
                f"Field {repr(h5_name)} should have dims {h5_dims}, but it is {value.ndim}-d."
            )

        expand_axes = []
        flip_axes = []
        layout_map = []
        slices = []
        for i, dim in enumerate(h5_dims_map):
            if dim == ExpandedDim:
                expand_axes.append(i)
                j = None
            elif dim in h5_dims:
                j = h5_dims.index(dim)
            elif -dim in h5_dims:
                j = h5_dims.index(-dim)
                flip_axes.append(j)
            else:
                raise ValueError(f"{dim} is not an HDF5 dim.")

            if j is not None:
                layout_map.append(j)

            slice_i = slice(0, None)
            if isinstance(dim, DataDim):
                slice_i = dim.index if dim.index is not None else slice_i
            slices.append(slice_i)

        value = np.flip(value, axis=flip_axes)
        value = np.transpose(value, axes=layout_map)
        value = np.expand_dims(value, axis=expand_axes)
        value = value[tuple(slices)]
        value = value.astype(self.gt4py_config.dtypes.from_name(dtype_name))

        ndim = len(dims)
        assert value.ndim == ndim

        return value

    def __del__(self) -> None:
        self.f.close()
