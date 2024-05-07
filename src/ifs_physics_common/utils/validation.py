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
from typing import Optional, TYPE_CHECKING
import warnings

from ifs_physics_common.utils.numpyx import to_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Tuple

    from sympl._core.data_array import DataArray
    from sympl._core.typingx import DataArrayDict


DEFAULT_ATOL: float = 1e-18
DEFAULT_RTOL: float = 1e-12


def get_storages_for_validation(field_a: DataArray, field_b: DataArray) -> Tuple[NDArray, NDArray]:
    a_np = to_numpy(field_a.data[...])
    b_np = to_numpy(field_b.data[...])
    slc = tuple(slice(0, min(s_src, s_trg)) for s_src, s_trg in zip(a_np.shape, b_np.shape))
    return a_np[slc], b_np[slc]


def validate(
    src: DataArrayDict,
    trg: DataArrayDict,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    common_keys = set(src.keys()).intersection(set(trg.keys()))
    for key in common_keys:
        src_field, trg_field = get_storages_for_validation(src[key], trg[key])
        assert src_field.shape == trg_field.shape

        if src_field.dtype.kind == "b":
            src_field = src_field.astype(float)
        if trg_field.dtype.kind == "b":
            trg_field = trg_field.astype(float)

        # remove nan's and inf's
        src_field = np.where(np.isnan(src_field), 0, src_field)
        src_field = np.where(np.isinf(src_field), 0, src_field)
        trg_field = np.where(np.isnan(trg_field), 0, trg_field)
        trg_field = np.where(np.isinf(trg_field), 0, trg_field)

        abs_diff = np.abs(src_field - trg_field)
        abs_diff_max = abs_diff.max()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            rel_diff = abs_diff / np.abs(trg_field)
        rel_diff_max = np.where(trg_field != 0, rel_diff, 0).max()

        atol = atol or DEFAULT_ATOL
        rtol = rtol or DEFAULT_RTOL
        allclose = np.allclose(src_field, trg_field, atol=atol, rtol=rtol, equal_nan=True)
        print(
            f"   - {key:20s}:"
            f"\033[9{2 if abs_diff_max < atol else 1}m max abs diff = {abs_diff_max:.5E}\033[00m,"
            f"\033[9{2 if rel_diff_max < rtol else 1}m max rel diff = {rel_diff_max:.5E}\033[00m,"
            f"\033[9{2 if allclose else 1}m allclose = {allclose}\033[00m"
        )
