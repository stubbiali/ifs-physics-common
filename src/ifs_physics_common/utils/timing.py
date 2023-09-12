# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sympl._core.time import Timer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Type


@contextmanager
def timing(label: str) -> Iterator[Type[Timer]]:
    try:
        Timer.start(label)
        yield Timer
    finally:
        Timer.stop()
