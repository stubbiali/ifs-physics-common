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

from collections.abc import Hashable, Sequence
import numpy as np
import numpy.typing as npt
from typing import TypeAlias, TypeVar, Union

from sympl._core.typingx import DataArray as SymplDataArray, DataArrayDict as SymplDataArrayDict

try:
    import cupy as cp
except ImportError:
    cp = np


DataArray: TypeAlias = SymplDataArray
DataArrayDict: TypeAlias = SymplDataArrayDict
NDArrayLike = Union[npt.NDArray, cp.ndarray]
NDArrayLikeDict = dict[str, NDArrayLike]
ParameterDict = dict[str, Union[bool, float, int]]
Property = dict[str, Union[str, Sequence[str], Hashable]]
PropertyDict = dict[str, Property]
Range = TypeVar("Range")
