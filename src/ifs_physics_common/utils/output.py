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
import csv
import datetime
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Literal, Optional, Sequence, Tuple


def write_performance_to_csv(
    output_file: str,
    host_name: str,
    precision: Literal["double", "single"],
    variant: str,
    num_cols: int,
    num_threads: int,
    nproma: int,
    num_runs: int,
    runtime_mean: float,
    runtime_stddev: float,
    mflops_mean: float,
    mflops_stddev: float,
) -> None:
    """Write performance statistics to a CSV file."""
    if not os.path.exists(output_file):
        with open(output_file, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(
                (
                    "date",
                    "host",
                    "precision",
                    "variant",
                    "num_cols",
                    "num_threads",
                    "nproma",
                    "num_runs",
                    "runtime_mean",
                    "runtime_stddev",
                    "mflops_mean",
                    "mflops_stddev",
                )
            )
    with open(output_file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(
            (
                datetime.date.today().strftime("%Y%m%d"),
                host_name,
                precision,
                variant,
                num_cols,
                num_threads,
                nproma,
                num_runs,
                runtime_mean,
                runtime_stddev,
                mflops_mean,
                mflops_stddev,
            )
        )


def write_stencils_performance_to_csv(
    output_file: str,
    host_name: str,
    precision: Literal["double", "single"],
    variant: str,
    num_cols: int,
    num_threads: int,
    num_runs: int,
    exec_info: Dict[str, Any],
    key_patterns: Sequence[str],
) -> None:
    call_time = 0.0
    for key, value in exec_info.items():
        if any(key_pattern in key for key_pattern in key_patterns):
            call_time += value["total_call_time"] * 1000 / num_runs

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(
                (
                    "date",
                    "host",
                    "precision",
                    "variant",
                    "num_cols",
                    "num_runs",
                    "num_threads",
                    "stencils",
                )
            )
    with open(output_file, "a") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            (
                datetime.date.today().strftime("%Y%m%d"),
                host_name,
                precision,
                variant,
                num_cols,
                num_runs,
                num_threads,
                call_time,
            )
        )


def print_performance(
    num_cols: int, runtime_l: Sequence[float], mflops_l: Optional[Sequence[float]] = None
) -> Tuple[float, float, float, float]:
    """Print means and standard deviation of runtimes and MFLOPS."""
    n = len(runtime_l)
    print(f"Performance over {num_cols} columns and {n} runs:")

    runtime_mean = sum(runtime_l) / n
    runtime_stddev = (
        sum((runtime - runtime_mean) ** 2 for runtime in runtime_l) / (n - 1 if n > 1 else n)
    ) ** 0.5
    print(f"-  Runtime: {runtime_mean:.3f} \u00B1 {runtime_stddev:.3f} ms.")

    mflops_l = mflops_l or [0.12482329 * num_cols / (runtime / 1000) for runtime in runtime_l]
    mflops_mean = sum(mflops_l) / n
    mflops_stddev = (
        sum((mflops - mflops_mean) ** 2 for mflops in mflops_l) / (n - 1 if n > 1 else n)
    ) ** 0.5
    print(f"-  MFLOPS: {mflops_mean:.3f} \u00B1 {mflops_stddev:.3f}.")

    return runtime_mean, runtime_stddev, mflops_mean, mflops_stddev
