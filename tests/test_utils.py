from ms_nexus_tools import lib as mnxlib


def test_count_digits():
    assert mnxlib.utils.count_digits(0) == 1

    assert mnxlib.utils.count_digits(1) == 1
    assert mnxlib.utils.count_digits(11) == 2
    assert mnxlib.utils.count_digits(111) == 3
    assert mnxlib.utils.count_digits(1111) == 4
    assert mnxlib.utils.count_digits(11111) == 5

    assert mnxlib.utils.count_digits(-1) == 1
    assert mnxlib.utils.count_digits(-11) == 2
    assert mnxlib.utils.count_digits(-111) == 3
    assert mnxlib.utils.count_digits(-1111) == 4
    assert mnxlib.utils.count_digits(-11111) == 5
