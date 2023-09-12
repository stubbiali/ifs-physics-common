# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

try:
    import cupy as cp
except ImportError:
    cp = np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ifs_physics_common.utils.typingx import ArrayLike


def to_numpy(storage: ArrayLike) -> NDArray:
    try:
        # storage is a cupy array
        return storage.get()  # type: ignore[no-any-return, union-attr]
    except AttributeError:
        return np.array(storage, copy=False)


def assign(lhs: ArrayLike, rhs: ArrayLike) -> None:
    if isinstance(lhs, cp.ndarray) and isinstance(rhs, np.ndarray):
        lhs[...] = cp.asarray(rhs)
    elif isinstance(lhs, np.ndarray) and isinstance(rhs, cp.ndarray):
        lhs[...] = rhs.get()
    else:
        lhs[...] = rhs
