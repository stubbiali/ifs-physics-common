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

try:
    import cupy as cp
except ImportError:
    cp = np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ifs_physics_common.typingx import NDArrayLike


def to_numpy(storage: NDArrayLike) -> NDArray:
    try:
        # storage is a cupy array
        return storage.get()  # type: ignore[no-any-return, union-attr]
    except AttributeError:
        return np.array(storage, copy=False)


def assign(lhs: NDArrayLike, rhs: NDArrayLike) -> None:
    if isinstance(lhs, cp.ndarray) and isinstance(rhs, np.ndarray):
        lhs[...] = cp.asarray(rhs)
    elif isinstance(lhs, np.ndarray) and isinstance(rhs, cp.ndarray):
        lhs[...] = rhs.get()
    else:
        lhs[...] = rhs
