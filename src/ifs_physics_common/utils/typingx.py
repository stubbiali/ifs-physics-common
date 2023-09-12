# -*- coding: utf-8 -*-
from collections.abc import Hashable, Sequence
import numpy as np
import numpy.typing as npt
from typing import Dict, TypeVar, Union

try:
    import cupy as cp
except ImportError:
    cp = np


ArrayLike = Union[npt.NDArray, cp.ndarray]
ArrayLikeDict = Dict[str, ArrayLike]
ParameterDict = Dict[str, Union[bool, float, int]]
Property = Dict[str, Union[str, Sequence[str], Hashable]]
PropertyDict = Dict[str, Property]
Range = TypeVar("Range")
