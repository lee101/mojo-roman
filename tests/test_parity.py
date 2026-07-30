import inspect
import re
import sys
from enum import IntEnum

import numpy as np
import pytest

import roman
from roman._lib import lib


PUBLISHED_VECTORS = (
    (0, "N"),
    (1, "I"),
    (3, "III"),
    (4, "IV"),
    (9, "IX"),
    (14, "XIV"),
    (19, "XIX"),
    (24, "XXIV"),
    (40, "XL"),
    (49, "XLIX"),
    (90, "XC"),
    (99, "XCIX"),
    (400, "CD"),
    (490, "CDXC"),
    (499, "CDXCIX"),
    (900, "CM"),
    (990, "CMXC"),
    (998, "CMXCVIII"),
    (999, "CMXCIX"),
    (2013, "MMXIII"),
)

INVALID_NUMERALS = (
    "",
    "Q12",
    "IIII",
    "VV",
    "VX",
    "IC",
    "IL",
    "IIV",
    "IXIX",
    "XXXX",
    "LC",
    "CDC",
    "DCD",
    "CMCM",
    "MMMMM",
    "MCMC",
    "NIX",
    " IX",
    "IX ",
    "Ⅳ",
    "ß",
)


def outcome(function, *args, **kwargs):
    try:
        return ("return", function(*args, **kwargs))
    except Exception as error:
        return ("raise", type(error).__name__, error.args)


@pytest.mark.parametrize(("number", "numeral"), PUBLISHED_VECTORS)
def test_published_to_roman_vectors(number, numeral):
    assert roman.toRoman(number) == numeral


@pytest.mark.parametrize(("number", "numeral"), PUBLISHED_VECTORS)
def test_published_from_roman_vectors(number, numeral):
    assert roman.fromRoman(numeral) == number


def test_to_roman_exhaustive_upstream_parity(upstream):
    assert [roman.toRoman(n) for n in range(5000)] == [
        upstream.toRoman(n) for n in range(5000)
    ]


def test_from_roman_exhaustive_upstream_parity(upstream):
    for number in range(5000):
        numeral = upstream.toRoman(number)
        assert roman.fromRoman(numeral) == upstream.fromRoman(numeral)


def test_scalar_cache_hit_paths():
    roman._cached_to_roman.cache_clear()
    roman._cached_from_roman.cache_clear()

    assert roman.toRoman(4999) == "MMMMCMXCIX"
    assert roman.toRoman(4999) == "MMMMCMXCIX"
    assert roman.fromRoman("mmmmcmxcix") == 4999
    assert roman.fromRoman("mmmmcmxcix") == 4999

    assert roman._cached_to_roman.cache_info().hits == 1
    assert roman._cached_from_roman.cache_info().hits == 1


def test_lowercase_exhaustive_upstream_parity(upstream):
    for number in range(5000):
        numeral = upstream.toRoman(number).lower()
        assert roman.fromRoman(numeral) == upstream.fromRoman(numeral)


@pytest.mark.parametrize("numeral", INVALID_NUMERALS)
@pytest.mark.parametrize("special_case", [True, False])
def test_invalid_numeral_upstream_parity(upstream, numeral, special_case):
    assert outcome(roman.fromRoman, numeral, special_case) == outcome(
        upstream.fromRoman, numeral, special_case
    )


@pytest.mark.parametrize("value", [-2, -1, 5000, 5001, 100000])
def test_to_roman_range_error_parity(upstream, value):
    assert outcome(roman.toRoman, value) == outcome(upstream.toRoman, value)


@pytest.mark.parametrize(
    "value", [1.0, 1.5, "1", complex(1), np.int64(1), None, [1]]
)
def test_to_roman_type_error_parity(upstream, value):
    assert outcome(roman.toRoman, value) == outcome(upstream.toRoman, value)


def test_bool_and_int_subclass_parity(upstream):
    class Number(IntEnum):
        FOUR = 4

    for value in (False, True, Number.FOUR):
        assert outcome(roman.toRoman, value) == outcome(upstream.toRoman, value)


@pytest.mark.parametrize("special_case", [True, 1, "yes", False, 0, None])
def test_zero_special_case_parity(upstream, special_case):
    for numeral in ("N", "n"):
        assert outcome(roman.fromRoman, numeral, special_case) == outcome(
            upstream.fromRoman, numeral, special_case
        )


@pytest.mark.parametrize("value", [None, 0, False, [], b"", 1, b"IX"])
def test_from_roman_non_string_parity(upstream, value):
    assert outcome(roman.fromRoman, value) == outcome(upstream.fromRoman, value)


def test_public_signatures_match_upstream(upstream):
    assert inspect.signature(roman.toRoman) == inspect.signature(upstream.toRoman)
    assert inspect.signature(roman.fromRoman) == inspect.signature(upstream.fromRoman)


def test_public_constants_match_upstream(upstream):
    assert roman.romanNumeralMap == upstream.romanNumeralMap
    probes = ["", "N", "I", "IIII", "MMMMCMXCIX", "MMMMM", "cmxliv"]
    for probe in probes:
        assert bool(roman.romanNumeralPattern.search(probe)) == bool(
            upstream.romanNumeralPattern.search(probe)
        )


def test_exception_hierarchy():
    assert issubclass(roman.OutOfRangeError, roman.RomanError)
    assert issubclass(roman.NotIntegerError, roman.RomanError)
    assert issubclass(roman.InvalidRomanNumeralError, roman.RomanError)


def test_every_output_is_canonical():
    canonical = re.compile(
        r"^(N|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))$"
    )
    assert all(canonical.fullmatch(roman.toRoman(n)) for n in range(5000))


def test_batch_to_roman_exhaustive(upstream):
    values = np.arange(5000, dtype=np.int64)
    assert roman.toRomanBatch(values) == [upstream.toRoman(int(n)) for n in values]


def test_batch_to_roman_accepts_integer_collections(upstream):
    values = [True, 4, np.int32(9), 4999]
    assert roman.toRomanBatch(values) == [
        upstream.toRoman(int(value)) for value in values
    ]


def test_batch_to_roman_empty():
    assert roman.toRomanBatch([]) == []
    assert roman.toRomanBatch(np.array([], dtype=np.int64)) == []


def test_batch_to_roman_handles_noncontiguous_and_wide_unsigned_inputs():
    source = np.arange(20, dtype=np.uint64)[::2]
    assert roman.toRomanBatch(source) == [roman.toRoman(int(n)) for n in source]
    with pytest.raises(roman.OutOfRangeError):
        roman.toRomanBatch(np.array([np.iinfo(np.uint64).max], dtype=np.uint64))


@pytest.mark.parametrize("values", [[1.5], ["I"], np.array([1.0])])
def test_batch_to_roman_rejects_nonintegers(values):
    with pytest.raises(roman.NotIntegerError):
        roman.toRomanBatch(values)


@pytest.mark.parametrize("values", [[-1, 1], [1, 5000]])
def test_batch_to_roman_rejects_out_of_range(values):
    with pytest.raises(roman.OutOfRangeError):
        roman.toRomanBatch(values)


def test_batch_from_roman_exhaustive(upstream):
    numerals = [upstream.toRoman(n).lower() for n in range(5000)]
    np.testing.assert_array_equal(
        roman.fromRomanBatch(numerals), np.arange(5000, dtype=np.int64)
    )


def test_batch_from_roman_special_case():
    np.testing.assert_array_equal(
        roman.fromRomanBatch(["N", "iv", "MMMMCMXCIX"]),
        np.array([0, 4, 4999]),
    )
    with pytest.raises(roman.InvalidRomanNumeralError):
        roman.fromRomanBatch(["N"], special_case=False)


def test_batch_from_roman_reports_bad_index():
    with pytest.raises(
        roman.InvalidRomanNumeralError,
        match=r"index 2: IIV",
    ):
        roman.fromRomanBatch(["I", "II", "IIV", "IV"])


def test_batch_from_roman_empty():
    result = roman.fromRomanBatch([])
    assert result.dtype == np.int64
    assert result.size == 0


def test_ffi_rejects_null_and_invalid_scalar_bounds():
    library = lib()
    assert library.mr_to_roman(1, 0, 17) < 0
    assert library.mr_to_roman(1, -1, 17) < 0
    assert library.mr_to_roman(1, 1, 14) < 0
    assert library.mr_to_roman(5000, 1, 17) < 0
    assert library.mr_from_roman(0, 1, 1, 1) < 0
    assert library.mr_from_roman(1, 2, 1, 1) < 0


def test_ffi_rejects_null_and_invalid_batch_bounds():
    library = lib()
    assert library.mr_to_roman_batch(0, 1, 1, 0, 16) < 0
    assert library.mr_to_roman_batch(-1, 1, 1, -1, 16) < 0
    assert library.mr_to_roman_batch(1, 2, 1, 1, 32) < 0
    assert library.mr_to_roman_batch(1, 1, 1, 1, 15) < 0
    assert library.mr_from_roman_batch(0, 1, 0, 2, 1, 1, 0, 1) < 0
    assert library.mr_from_roman_batch(1, 1, 1, 1, 1, 1, 1, 1) < 0


def test_parse_args_parity(monkeypatch, upstream):
    for argv in (["roman", "10"], ["roman", "--reverse", "x"]):
        monkeypatch.setattr(sys, "argv", argv)
        ours = roman.parse_args()
        theirs = upstream.parse_args()
        assert vars(ours) == vars(theirs)


def test_main_to_roman(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["roman", "944"])
    assert roman.main() == 0
    assert capsys.readouterr().out == "CMXLIV\n"


def test_main_from_roman(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["roman", "--reverse", "cmxliv"])
    assert roman.main() == 0
    assert capsys.readouterr().out == "944\n"
