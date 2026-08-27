# mojo-roman

`mojo-roman` is a Mojo implementation of the
[`roman`](https://pypi.org/project/roman/) 5.2 numeral converter, exposed to
Python through a drop-in `roman` module.

The scalar API matches upstream:

- `toRoman(n: int) -> str`
- `fromRoman(s: str, special_case: bool = True) -> int`
- `RomanError`, `OutOfRangeError`, `NotIntegerError`, and
  `InvalidRomanNumeralError`
- `romanNumeralMap`, `romanNumeralPattern`, `parse_args()`, and `main()`

It accepts zero as `N`, canonical numerals through 4999, and case-insensitive
input just like upstream. `toRomanBatch()` and `fromRomanBatch()` are additive
bulk APIs that amortize the Python-to-Mojo call cost.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-roman.so`.

## Usage

```python
import numpy as np
import roman

assert roman.toRoman(944) == "CMXLIV"
assert roman.fromRoman("cmxliv") == 944
assert roman.toRoman(0) == "N"

values = np.array([0, 4, 944, 4999], dtype=np.int64)
numerals = roman.toRomanBatch(values)
assert numerals == ["N", "IV", "CMXLIV", "MMMMCMXCIX"]

decoded = roman.fromRomanBatch(numerals)
np.testing.assert_array_equal(decoded, values)
```

The module also runs as a command:

```bash
pixi run python -m roman 972
pixi run python -m roman --reverse cMlxxii
```

## Coverage and limits

The conversion-facing API of upstream `roman` 5.2 is covered: `toRoman`,
`fromRoman`, the four exception classes, the numeral map and validation
pattern, `parse_args`, and `main`. Tests compare against the installed
conda-forge package across the full supported integer range and its canonical
lowercase numerals. Separate tests cover invalid forms, error types and
messages, signatures, constants, and CLI helpers.

This library deliberately does not add numeral systems that upstream does not
support: there are no overlines, parenthesized thousands, fractions,
non-canonical spellings, negative values, or values above 4999. The batch
functions are extensions rather than upstream APIs; `fromRomanBatch()` returns
a contiguous NumPy `int64` array.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, and Python 3.13.14:

| case | mojo-roman | upstream roman 5.2 | speedup |
| --- | ---: | ---: | ---: |
| `toRoman` scalar loop (100k) | 27.60 ms | 120.41 ms | 4.36x |
| `fromRoman` scalar loop (100k) | 38.16 ms | 298.89 ms | 7.83x |
| `toRomanBatch` (1M) | 124.56 ms | 1277.70 ms | 10.26x |
| `fromRomanBatch` (1M) | 94.63 ms | 3052.72 ms | 32.26x |

These are the best of three repetitions from an actual `pixi run bench` run.
The task takes a machine-wide lock before measuring. Scalar results include
bounded cache hits; batch results execute one native call per batch. The
conversion kernels are branch-heavy operations on strings of at most 15 bytes
with no floating-point arithmetic, so they do not approach the roughly two
flops-per-byte threshold needed to justify device transfer and launch costs.
The library therefore has no GPU path. MAX is used only for CPU parallelism.

## How it works

`src/roman.mojo` is one compilation unit containing decimal-place encoding,
canonical grammar validation, and four C ABI exports. Python calls the shared
library with `ctypes` on scalar cache misses and for every bulk call. Buffers
cross the ABI as integer addresses and are rebuilt in Mojo as
`UnsafePointer[..., AnyOrigin[mut=True]]`, avoiding parametric exported
functions.

Python owns every allocation. A scalar encoder writes into a fixed-size buffer.
The bulk encoder writes newline-delimited ASCII into one preallocated buffer,
which Python decodes in one operation. Bulk parsing packs the source strings
into one byte buffer and passes an `int64` offset table; Mojo writes results
directly into a contiguous NumPy array. Mojo allocates no heap memory, and no
buffer outlives the call. Each export validates non-null addresses, lengths,
capacities, counts, value ranges, and offset ordering before accessing memory;
Python keeps every backing object alive for the duration of the `ctypes` call.

Bulk parsing builds its packed ASCII buffer in one encode and computes offsets
with NumPy. Mojo validates offsets and scans parse results with SIMD, including
scalar remainder loops for non-multiple lengths. Batches below 65,536 items stay
serial; larger batches are split into independent 8,192-item CPU tasks.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task takes a machine-wide flock before measuring. Benchmark
figures above are recorded output, including observed timing regressions.

MIT.
