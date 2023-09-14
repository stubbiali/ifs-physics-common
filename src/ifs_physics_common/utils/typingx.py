# -*- coding: utf-8 -*-
from collections.abc import Hashable, Sequence
import numpy as np
import numpy.typing as npt
from typing import Dict, TypeAlias, TypeVar, Union

from sympl._core.typingx import DataArray as SymplDataArray, DataArrayDict as SymplDataArrayDict

try:
    import cupy as cp
except ImportError:
    cp = np


DataArray: TypeAlias = SymplDataArray
DataArrayDict: TypeAlias = SymplDataArrayDict
NDArrayLike = Union[npt.NDArray, cp.ndarray]
NDArrayLikeDict = Dict[str, NDArrayLike]
ParameterDict = Dict[str, Union[bool, float, int]]
Property = Dict[str, Union[str, Sequence[str], Hashable]]
PropertyDict = Dict[str, Property]
Range = TypeVar("Range")
