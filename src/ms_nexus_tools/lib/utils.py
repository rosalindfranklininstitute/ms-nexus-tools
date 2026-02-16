def count_digits(num: int) -> int:
    digits = 1
    num = abs(num) // 10
    while abs(num) > 0:
        digits += 1
        num = num // 10
    return digits
