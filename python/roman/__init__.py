"""Convert to and from Roman numerals using Mojo kernels."""

import argparse
import ctypes
import re
import sys
from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from ._lib import addr, bytes_addr, lib


__author__ = "Mark Pilgrim (f8dy@diveintopython.org)"
__copyright__ = """Copyright (c) 2001 Mark Pilgrim

This program is part of "Dive Into Python", a free Python tutorial for
experienced programmers.  Visit http://diveintopython.org/ for the
latest version.
"""


class RomanError(Exception):
    pass


class OutOfRangeError(RomanError):
    pass


class NotIntegerError(RomanError):
    pass


class InvalidRomanNumeralError(RomanError):
    pass


romanNumeralMap = (
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
)

romanNumeralPattern = re.compile(
    r"""
    ^
    M{0,4}
    (CM|CD|D?C{0,3})
    (XC|XL|L?X{0,3})
    (IX|IV|V?I{0,3})
    $
    """,
    re.VERBOSE,
)


@lru_cache(maxsize=5000)
def _cached_to_roman(n: int) -> str:
    buffer = ctypes.create_string_buffer(17)
    length = lib().mr_to_roman(n, ctypes.addressof(buffer), len(buffer))
    if not 0 < length < len(buffer):
        raise RuntimeError("Mojo encoder rejected a validated scalar call")
    return buffer.raw[:length].decode("ascii")


def toRoman(n: int) -> str:
    """Convert an integer in the inclusive range 0..4999 to a Roman numeral."""
    if not isinstance(n, int):
        raise NotIntegerError("decimals cannot be converted")
    if not (-1 < n < 5000):
        raise OutOfRangeError("number out of range (must be 0..4999)")
    return _cached_to_roman(n)


@lru_cache(maxsize=10000)
def _cached_from_roman(upper: str, special_case: bool) -> int:
    if upper == "N" and special_case:
        return 0
    try:
        encoded = upper.encode("ascii")
    except UnicodeEncodeError:
        raise InvalidRomanNumeralError(
            "Invalid Roman numeral: %s" % upper
        ) from None

    result = lib().mr_from_roman(
        bytes_addr(encoded), len(encoded), len(encoded), int(bool(special_case))
    )
    if result == -2:
        raise RuntimeError("Mojo parser rejected a validated scalar call")
    if result < 0:
        raise InvalidRomanNumeralError("Invalid Roman numeral: %s" % upper)
    return result


def fromRoman(s: str, special_case: bool = True) -> int:
    """Convert a canonical Roman numeral to an integer."""
    if not s:
        raise InvalidRomanNumeralError("Input cannot be blank")
    if type(s) is str:
        upper = s.upper()
        if upper == "N" and special_case:
            return 0
        return _cached_from_roman(upper, bool(special_case))

    upper = s.upper()
    if upper == "N" and special_case:
        return 0
    if not isinstance(upper, str):
        romanNumeralPattern.search(upper)
    try:
        encoded = upper.encode("ascii")
    except UnicodeEncodeError:
        raise InvalidRomanNumeralError(
            "Invalid Roman numeral: %s" % upper
        ) from None

    result = lib().mr_from_roman(
        bytes_addr(encoded), len(encoded), len(encoded), int(bool(special_case))
    )
    if result == -2:
        raise RuntimeError("Mojo parser rejected a validated scalar call")
    if result < 0:
        raise InvalidRomanNumeralError("Invalid Roman numeral: %s" % upper)
    return result


def toRomanBatch(numbers: Iterable[int]) -> list[str]:
    """Convert a one-dimensional collection in one Mojo call."""
    if isinstance(numbers, np.ndarray):
        source = numbers
    else:
        source = np.asarray(list(numbers))
    if source.ndim != 1:
        raise NotIntegerError("decimals cannot be converted")
    if source.size == 0:
        return []
    if source.dtype.kind not in "iub":
        raise NotIntegerError("decimals cannot be converted")
    if np.any(source < 0) or np.any(source >= 5000):
        raise OutOfRangeError("number out of range (must be 0..4999)")

    values = np.ascontiguousarray(source, dtype=np.int64)
    buffer = ctypes.create_string_buffer(17 * values.size)
    used = lib().mr_to_roman_batch(
        addr(values),
        values.size,
        values.size,
        ctypes.addressof(buffer),
        len(buffer),
    )
    if used < 0 or used > len(buffer):
        raise RuntimeError("Mojo encoder rejected a validated batch call")
    return buffer.raw[:used].decode("ascii").splitlines()


def fromRomanBatch(
    numerals: Iterable[str], special_case: bool = True
) -> np.ndarray:
    """Convert a collection to a contiguous int64 NumPy array in one Mojo call."""
    items = list(numerals)
    if not items:
        return np.empty(0, dtype=np.int64)

    try:
        lengths = np.fromiter(map(len, items), dtype=np.int64, count=len(items))
        packed = "".join(items).encode("ascii")
    except (TypeError, UnicodeEncodeError):
        for index, item in enumerate(items):
            if not isinstance(item, str):
                raise InvalidRomanNumeralError(
                    f"Invalid Roman numeral at index {index}: {item}"
                ) from None
            try:
                item.encode("ascii")
            except UnicodeEncodeError:
                raise InvalidRomanNumeralError(
                    f"Invalid Roman numeral at index {index}: {item}"
                ) from None
        raise RuntimeError("failed to pack validated Roman numerals")
    empty = np.flatnonzero(lengths == 0)
    if empty.size:
        index = int(empty[0])
        raise InvalidRomanNumeralError(
            f"Invalid Roman numeral at index {index}: {items[index]}"
        )

    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    result = np.empty(len(items), dtype=np.int64)
    failed = lib().mr_from_roman_batch(
        bytes_addr(packed),
        len(packed),
        addr(offsets),
        offsets.size,
        len(items),
        int(bool(special_case)),
        addr(result),
        result.size,
    )
    if failed < 0:
        raise RuntimeError("Mojo parser rejected a validated batch call")
    if failed:
        index = failed - 1
        raise InvalidRomanNumeralError(
            f"Invalid Roman numeral at index {index}: {items[index].upper()}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roman", description="convert between roman and arabic numerals"
    )
    parser.add_argument("number", help="the value to convert")
    parser.add_argument(
        "-r",
        "--reverse",
        action="store_true",
        default=False,
        help="convert roman to numeral (case insensitive) [default: False]",
    )
    args = parser.parse_args()
    args.number = args.number
    return args


def main() -> int:
    args = parse_args()
    if args.reverse:
        print(fromRoman(args.number))
    else:
        print(toRoman(int(args.number)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
