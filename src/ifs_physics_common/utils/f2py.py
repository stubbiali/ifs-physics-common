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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Optional, Union


PORTED_OBJECTS = {}


def ported_object(
    handle: Optional[Any] = None,
    from_file: Optional[Union[str, Sequence[str]]] = None,
    from_line: Optional[int] = None,
    to_line: Optional[int] = None,
) -> Union[Any, Callable[[Any], Any]]:
    if from_line is not None and to_line is not None:
        assert from_line <= to_line

    def core(obj: Any) -> Any:
        PORTED_OBJECTS[obj.__name__] = obj
        setattr(obj, "from_file", from_file)
        setattr(obj, "from_line", from_line)
        setattr(obj, "to_line", to_line)
        return obj

    if handle is not None:
        return core(handle)
    else:
        return core


# convenient aliases to improve readability
ported_class = ported_object
ported_function = ported_object
ported_method = ported_object
