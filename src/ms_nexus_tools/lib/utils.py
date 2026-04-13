def count_digits(num: int) -> int:
    """
    Counts the number of digits in an integer:
    >>> count_digits(1), count_digits(2)
    (1, 1)

    >>count_digits(10), count_digits(12)
    (2, 2)

    >>> count_digits(0)
    1
    """
    digits = 1
    num = abs(num) // 10
    while abs(num) > 0:
        digits += 1
        num = num // 10
    return digits


def slice_len(slc: slice) -> int:
    """
    Returns the length of a slice
    >>> slice_len(slice(5))
    5

    >>> slice_len(slice(1, 5))
    4

    >>> slice_len(slice(1, 5, 2))
    2
    """
    inc = slc.step or 1

    if slc.start is None:
        return slc.stop // inc
    else:
        return (slc.stop - slc.start) // inc
