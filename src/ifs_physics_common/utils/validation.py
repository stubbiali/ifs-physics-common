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
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.utils.numpyx import to_numpy

if TYPE_CHECKING:
    from typing import Tuple

    from sympl._core.data_array import DataArray
    from sympl._core.typingx import DataArrayDict

    from ifs_physics_common.utils.typingx import NDArrayLike


def validate_storage_2d(src: NDArrayLike, trg: NDArrayLike) -> bool:
    src_np = to_numpy(src)
    trg_np = to_numpy(trg)
    mi = min(src_np.shape[0], trg_np.shape[0])
    mj = min(src_np.shape[1], trg_np.shape[1])
    return np.allclose(src_np[:mi, :mj], trg_np[:mi, :mj], atol=1e-18, rtol=1e-12)


def validate_storage_3d(src: NDArrayLike, trg: NDArrayLike) -> bool:
    src_np = to_numpy(src)
    trg_np = to_numpy(trg)
    mi = min(src_np.shape[0], trg_np.shape[0])
    mj = min(src_np.shape[1], trg_np.shape[1])
    mk = min(src_np.shape[2], trg_np.shape[2])
    return np.allclose(src_np[:mi, :mj, :mk], trg_np[:mi, :mj, :mk], atol=1e-18, rtol=1e-12)


def validate_field(src: DataArray, trg: DataArray) -> bool:
    if src.ndim == 2:
        return validate_storage_2d(src.data, trg.data)
    elif src.ndim == 3:
        return validate_storage_3d(src.data, trg.data)
    else:
        raise ValueError("The field to validate must be either 2-d or 3-d.")


def validate(src: DataArrayDict, trg: DataArrayDict) -> Tuple[str, ...]:
    return tuple(
        name
        for name in src
        if name in trg and name != "time" and not validate_field(src[name], trg[name])
    )
