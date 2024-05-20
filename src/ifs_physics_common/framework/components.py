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
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from sympl._core.core_components import (
    DiagnosticComponent as SymplDiagnosticComponent,
    ImplicitTendencyComponent as SymplImplicitTendencyComponent,
)

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.storage import gt_zeros

if TYPE_CHECKING:
    from typing import Any, Optional

    from gt4py.cartesian import StencilObject
    from sympl._core.typingx import PropertyDict

    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import NDArrayLike


class ComputationalGridComponent:
    """Model component defined over a computational grid."""

    def __init__(self, computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig) -> None:
        self.computational_grid = computational_grid
        self.gt4py_config = gt4py_config

    def compile_stencil(
        self, name: str, externals: Optional[dict[str, Any]] = None
    ) -> StencilObject:
        return compile_stencil(name, self.gt4py_config, externals)

    def fill_properties_with_dims(self, properties: PropertyDict) -> PropertyDict:
        for field_name, field_prop in properties.items():
            assert "grid_dims" in field_prop
            dim_names = self.computational_grid.grids[field_prop["grid_dims"]].dim_names
            data_dims = field_prop.get("data_dims", ())
            field_prop["dims"] = dim_names + tuple(str(dim) for dim in data_dims)
        return properties

    def allocate(self, name: str, properties: PropertyDict) -> NDArrayLike:
        data_dims = properties.get("data_dims", ())
        dtype_name = properties.get("dtype_name", "float")
        return gt_zeros(
            self.computational_grid,
            properties[name]["grid_dims"],
            data_dims=data_dims,
            gt4py_config=self.gt4py_config,
            dtype_name=dtype_name,
        )


class DiagnosticComponent(ComputationalGridComponent, SymplDiagnosticComponent):
    """Grid-aware variant of Sympl's ``DiagnosticComponent``."""

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(computational_grid, gt4py_config=gt4py_config)
        super(ComputationalGridComponent, self).__init__(enable_checks=enable_checks)

    @cached_property
    def input_properties(self) -> PropertyDict:
        return self.fill_properties_with_dims(self.input_grid_properties)

    @abstractmethod
    @cached_property
    def input_grid_properties(self) -> PropertyDict:
        """
        Dictionary where each key is the name of an input field, and the corresponding value is a
        dictionary specifying the units for that field ('units') and the identifier of the grid over
        which it is defined ('grid').
        """
        ...

    def allocate_diagnostic(self, name: str) -> NDArrayLike:
        return self.allocate(name, self.diagnostic_properties)

    @cached_property
    def diagnostic_properties(self) -> PropertyDict:
        return self.fill_properties_with_dims(self.diagnostic_grid_properties)

    @abstractmethod
    @cached_property
    def diagnostic_grid_properties(self) -> PropertyDict:
        """
        Dictionary where each key is the name of a field diagnosed by the component, and the
        corresponding value is a dictionary specifying the units for that field ('units') and the
        identifier of the grid over which it is defined ('grid').
        """
        ...


class ImplicitTendencyComponent(ComputationalGridComponent, SymplImplicitTendencyComponent):
    """Grid-aware variant of Sympl's ``ImplicitTendencyComponent``."""

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(computational_grid, gt4py_config=gt4py_config)
        super(ComputationalGridComponent, self).__init__(enable_checks=enable_checks)

    @cached_property
    def input_properties(self) -> PropertyDict:
        return self.fill_properties_with_dims(self.input_grid_properties)

    @abstractmethod
    @cached_property
    def input_grid_properties(self) -> PropertyDict:
        """
        Dictionary where each key is the name of an input field, and the corresponding value is a
        dictionary specifying the units for that field ('units') and the identifier of the grid over
        which it is defined ('grid').
        """
        ...

    def allocate_tendency(self, name: str) -> NDArrayLike:
        return self.allocate(name, self.tendency_properties)

    @cached_property
    def tendency_properties(self) -> PropertyDict:
        return self.fill_properties_with_dims(self.tendency_grid_properties)

    @abstractmethod
    @cached_property
    def tendency_grid_properties(self) -> PropertyDict:
        """
        Dictionary where each key is the name of a tendency field computed by the component, and the
        corresponding value is a dictionary specifying the units for that field ('units') and the
        identifier of the grid over which it is defined ('grid').
        """
        ...

    def allocate_diagnostic(self, name: str) -> NDArrayLike:
        return self.allocate(name, self.diagnostic_properties)

    @cached_property
    def diagnostic_properties(self) -> PropertyDict:
        return self.fill_properties_with_dims(self.diagnostic_grid_properties)

    @abstractmethod
    @cached_property
    def diagnostic_grid_properties(self) -> PropertyDict:
        """
        Dictionary where each key is the name of a field diagnosed by the component, and the
        corresponding value is a dictionary specifying the units for that field ('units') and the
        identifier of the grid over which it is defined ('grid').
        """
        ...
