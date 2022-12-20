import time

from imprint.timer import new_mock_timer
from imprint.timer import Timer


def test_timer_zero():
    t = Timer()
    start = int(time.time())
    assert t.unique() == start
    assert t.unique() == start + 1
    assert t.unique() == start + 2


def test_timer_mock():
    t = new_mock_timer()
    assert t.unique() == 0
    assert t.unique() == 1
    assert t.unique() == 2
