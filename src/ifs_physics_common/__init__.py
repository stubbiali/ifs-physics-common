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

from ifs_physics_common.framework.components import (
    ComputationalGridComponent,
    DiagnosticComponent,
    ImplicitTendencyComponent,
)
from ifs_physics_common.framework.config import (
    DataTypes,
    FortranConfig,
    GT4PyConfig,
    IOConfig,
    PythonConfig,
)
from ifs_physics_common.framework.grid import ComputationalGrid, Grid
from ifs_physics_common.framework.stencil import (
    compile_stencil,
    function_collection,
    stencil_collection,
)
from ifs_physics_common.framework.storage import (
    allocate_data_array,
    get_data_array,
    get_data_shape_from_name,
    get_dtype_from_name,
    managed_temporary_storage,
    managed_temporary_storage_pool,
    zeros,
)
from ifs_physics_common.utils.f2py import ported_class, ported_function, ported_method
from ifs_physics_common.utils.numpyx import assign, to_numpy
from ifs_physics_common.utils.output import (
    print_performance,
    write_performance_to_csv,
    write_stencils_performance_to_csv,
)
from ifs_physics_common.utils.timing import timing
from ifs_physics_common.utils.validation import (
    get_storages_for_validation,
    validate,
    validate_field,
)

__version__ = "0.3.0.dev"

__all__ = [
    "ComputationalGrid",
    "ComputationalGridComponent",
    "DataTypes",
    "DiagnosticComponent",
    "FortranConfig",
    "GT4PyConfig",
    "Grid",
    "IOConfig",
    "ImplicitTendencyComponent",
    "PythonConfig",
    "__version__",
    "allocate_data_array",
    "assign",
    "compile_stencil",
    "function_collection",
    "get_data_array",
    "get_data_shape_from_name",
    "get_dtype_from_name",
    "get_storages_for_validation",
    "managed_temporary_storage",
    "managed_temporary_storage_pool",
    "ported_class",
    "ported_function",
    "ported_method",
    "print_performance",
    "stencil_collection",
    "timing",
    "to_numpy",
    "validate",
    "validate_field",
    "write_performance_to_csv",
    "write_stencils_performance_to_csv",
    "zeros",
]
