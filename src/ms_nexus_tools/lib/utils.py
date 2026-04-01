def count_digits(num: int) -> int:
    digits = 1
    num = abs(num) // 10
    while abs(num) > 0:
        digits += 1
        num = num // 10
    return digits


def slice_len(slc: slice) -> int:
    inc = slc.step or 1

    if slc.start is None:
        return slc.stop // inc
    else:
        return (slc.stop - slc.start) // inc
