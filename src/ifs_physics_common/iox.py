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

from ifs_physics_common.grid import DataDim, ExpandedDim
from ifs_physics_common.storage import assign, zeros

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray
    from pydantic import BaseModel
    from typing import Literal, Optional, Union

    from sympl._core.typingx import DataArray

    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import (
        AbstractGridDimTuple,
        ComputationalGrid,
        DataDimTuple,
        DimTuple,
    )


class HDF5Operator:
    f: File
    gt4py_config: GT4PyConfig

    def __init__(self, filename: str, *, gt4py_config: GT4PyConfig) -> None:
        self.f = File(filename, mode="r")
        self.gt4py_config = gt4py_config

    def __del__(self) -> None:
        self.f.close()

    def get_params(
        self, param_cls: type[BaseModel], get_param_name: Optional[Callable[[str], str]] = None
    ) -> BaseModel:
        init_dict: dict[str, Union[bool, float, int]] = {}
        for attr_name, metadata in param_cls.schema()["properties"].items():
            param_name = get_param_name(attr_name) if get_param_name is not None else attr_name
            param_type = metadata["type"]
            param_default = metadata.get("default")
            if param_type == "boolean":
                init_dict[attr_name] = self.gt4py_config.dtypes.bool(
                    self.f.get(param_name, [param_default if param_default is not None else True])[
                        0
                    ]
                )
            elif param_type == "number":
                init_dict[attr_name] = self.gt4py_config.dtypes.float(
                    self.f.get(param_name, [param_default if param_default is not None else 0.0])[0]
                )
            elif param_type == "integer":
                init_dict[attr_name] = self.gt4py_config.dtypes.int(
                    self.f.get(param_name, [param_default if param_default is not None else 0])[0]
                )
            else:
                raise ValueError(f"Invalid parameter type `{param_type}`.")
        return param_cls(**init_dict)


class HDF5GridOperator:
    computational_grid: ComputationalGrid
    f: File
    gt4py_config: GT4PyConfig
    hdf5_operator: HDF5Operator

    def __init__(
        self, filename: str, computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
    ) -> None:
        self.computational_grid = computational_grid
        self.hdf5_operator = HDF5Operator(filename, gt4py_config=gt4py_config)

        # convenient shortcuts
        self.f = self.hdf5_operator.f
        self.gt4py_config = self.hdf5_operator.gt4py_config

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
        rhs = self._read_raw_field(grid_dims + data_dims, dtype_name, h5_name, h5_dims, h5_dims_map)
        assign(field, rhs)
        return field

    def _read_raw_field(
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
