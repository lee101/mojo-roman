"""Roman numeral conversion kernels and their C ABI."""

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def write_repeated(dst: BPtr, pos: Int, value: UInt8, count: Int) -> Int:
    var next_pos = pos
    for _ in range(count):
        dst[next_pos] = value
        next_pos += 1
    return next_pos


def write_digit(
    dst: BPtr,
    pos: Int,
    digit: Int,
    one: UInt8,
    five: UInt8,
    ten: UInt8,
) -> Int:
    var next_pos = pos
    if digit == 9:
        dst[next_pos] = one
        dst[next_pos + 1] = ten
        return next_pos + 2
    if digit >= 5:
        dst[next_pos] = five
        next_pos += 1
        return write_repeated(dst, next_pos, one, digit - 5)
    if digit == 4:
        dst[next_pos] = one
        dst[next_pos + 1] = five
        return next_pos + 2
    return write_repeated(dst, next_pos, one, digit)


def to_roman(n: Int, dst: BPtr) -> Int:
    if n == 0:
        dst[0] = UInt8(78)
        return 1

    var value = n
    var pos = write_repeated(dst, 0, UInt8(77), value // 1000)
    value %= 1000
    pos = write_digit(
        dst, pos, value // 100, UInt8(67), UInt8(68), UInt8(77)
    )
    value %= 100
    pos = write_digit(
        dst, pos, value // 10, UInt8(88), UInt8(76), UInt8(67)
    )
    pos = write_digit(
        dst, pos, value % 10, UInt8(73), UInt8(86), UInt8(88)
    )
    return pos


def upper_ascii(value: UInt8) -> UInt8:
    if value >= UInt8(97) and value <= UInt8(122):
        return value - UInt8(32)
    return value


def matches(src: BPtr, index: Int, n: Int, value: UInt8) -> Bool:
    return index < n and upper_ascii(src[index]) == value


def matches_pair(
    src: BPtr, index: Int, n: Int, first: UInt8, second: UInt8
) -> Bool:
    return (
        index + 1 < n
        and upper_ascii(src[index]) == first
        and upper_ascii(src[index + 1]) == second
    )


def from_roman(src: BPtr, n: Int, special_case: Bool) -> Int:
    if n == 0:
        return -1
    if special_case and n == 1 and upper_ascii(src[0]) == UInt8(78):
        return 0

    var index = 0
    var result = 0
    var count = 0

    while count < 4 and matches(src, index, n, UInt8(77)):
        result += 1000
        index += 1
        count += 1

    if matches_pair(src, index, n, UInt8(67), UInt8(77)):
        result += 900
        index += 2
    elif matches_pair(src, index, n, UInt8(67), UInt8(68)):
        result += 400
        index += 2
    else:
        if matches(src, index, n, UInt8(68)):
            result += 500
            index += 1
        count = 0
        while count < 3 and matches(src, index, n, UInt8(67)):
            result += 100
            index += 1
            count += 1

    if matches_pair(src, index, n, UInt8(88), UInt8(67)):
        result += 90
        index += 2
    elif matches_pair(src, index, n, UInt8(88), UInt8(76)):
        result += 40
        index += 2
    else:
        if matches(src, index, n, UInt8(76)):
            result += 50
            index += 1
        count = 0
        while count < 3 and matches(src, index, n, UInt8(88)):
            result += 10
            index += 1
            count += 1

    if matches_pair(src, index, n, UInt8(73), UInt8(88)):
        result += 9
        index += 2
    elif matches_pair(src, index, n, UInt8(73), UInt8(86)):
        result += 4
        index += 2
    else:
        if matches(src, index, n, UInt8(86)):
            result += 5
            index += 1
        count = 0
        while count < 3 and matches(src, index, n, UInt8(73)):
            result += 1
            index += 1
            count += 1

    if index != n:
        return -1
    return result


@export("mr_to_roman")
def mr_to_roman(n: Int, dst_addr: Int, dst_capacity: Int) abi("C") -> Int:
    if n < 0 or n >= 5000 or dst_addr <= 0 or dst_capacity < 15:
        return -1
    return to_roman(n, BPtr(unsafe_from_address=dst_addr))


@export("mr_from_roman")
def mr_from_roman(
    src_addr: Int, n: Int, src_capacity: Int, special_case: Int
) abi("C") -> Int:
    if src_addr <= 0 or n < 0 or n > src_capacity:
        return -2
    return from_roman(
        BPtr(unsafe_from_address=src_addr), n, special_case != 0
    )


@export("mr_to_roman_batch")
def mr_to_roman_batch(
    values_addr: Int,
    count: Int,
    values_count: Int,
    dst_addr: Int,
    dst_capacity: Int,
) abi("C") -> Int:
    if count < 0 or count > values_count:
        return -1
    if count == 0:
        return 0
    if (
        values_addr <= 0
        or dst_addr <= 0
        or dst_capacity < 0
        or count > dst_capacity // 16
    ):
        return -1
    var values = IPtr(unsafe_from_address=values_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(count):
        if values[i] < 0 or values[i] >= 5000:
            return -2
    var pos = 0
    for i in range(count):
        pos += to_roman(Int(values[i]), dst + pos)
        dst[pos] = UInt8(10)
        pos += 1
    return pos


@export("mr_from_roman_batch")
def mr_from_roman_batch(
    data_addr: Int,
    data_length: Int,
    offsets_addr: Int,
    offsets_count: Int,
    count: Int,
    special_case: Int,
    dst_addr: Int,
    dst_count: Int,
) abi("C") -> Int:
    if count < 0 or offsets_count != count + 1 or dst_count < count:
        return -1
    if count == 0:
        return 0
    if (
        data_addr <= 0
        or offsets_addr <= 0
        or dst_addr <= 0
        or data_length <= 0
    ):
        return -1
    var data = BPtr(unsafe_from_address=data_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)
    if offsets[0] != 0 or Int(offsets[count]) != data_length:
        return -1
    for i in range(count):
        var start = Int(offsets[i])
        var end = Int(offsets[i + 1])
        if start < 0 or end <= start or end > data_length:
            return -1
        var value = from_roman(
            data + start,
            end - start,
            special_case != 0,
        )
        if value < 0:
            return i + 1
        dst[i] = Int64(value)
    return 0
