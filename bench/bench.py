"""Benchmark mojo-roman against upstream roman 5.2."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import math
import os
import platform
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import roman  # noqa: E402


def load_upstream():
    distribution = importlib.metadata.distribution("roman")
    source = distribution.locate_file("roman/__init__.py")
    spec = importlib.util.spec_from_file_location("_benchmark_upstream_roman", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
        del result
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    upstream = load_upstream()
    scalar_values = np.arange(100_000, dtype=np.int64) % 5000
    batch_values = np.arange(1_000_000, dtype=np.int64) % 5000
    scalar_numerals = [upstream.toRoman(int(n)) for n in scalar_values]
    batch_numerals = [upstream.toRoman(int(n)) for n in batch_values]

    cases = [
        (
            "toRoman scalar loop (100k)",
            lambda: [roman.toRoman(int(n)) for n in scalar_values],
            lambda: [upstream.toRoman(int(n)) for n in scalar_values],
        ),
        (
            "fromRoman scalar loop (100k)",
            lambda: [roman.fromRoman(s) for s in scalar_numerals],
            lambda: [upstream.fromRoman(s) for s in scalar_numerals],
        ),
        (
            "toRomanBatch (1M)",
            lambda: roman.toRomanBatch(batch_values),
            lambda: [upstream.toRoman(int(n)) for n in batch_values],
        ),
        (
            "fromRomanBatch (1M)",
            lambda: roman.fromRomanBatch(batch_numerals),
            lambda: [upstream.fromRoman(s) for s in batch_numerals],
        ),
    ]

    roman.toRoman(1)
    assert roman.toRomanBatch(batch_values[:5000]) == batch_numerals[:5000]
    np.testing.assert_array_equal(
        roman.fromRomanBatch(batch_numerals[:5000]), batch_values[:5000]
    )

    print(
        f"Machine: {cpu_name()}, {platform.system()} "
        f"{platform.release()}, Python {platform.python_version()}"
    )
    print()
    print("| case | mojo-roman | upstream roman 5.2 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function in cases:
        mojo_seconds = timeit(mojo_function)
        upstream_seconds = timeit(upstream_function)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
