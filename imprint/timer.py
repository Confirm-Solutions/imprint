"""
This module makes it easy to mock the non-repeating timer function used by
imprint so that test results can be made reproducible.

non-repeating: The timer function will never return the same value twice even
if the time has not changed.

To replace the timer with the incrementing mock, use the following code:

with mock.patch("imprint.timer._timer", ip.timer.new_mock_timer()):
    ...

or

@mock.patch("imprint.timer._timer", ip.timer.new_mock_timer())
def test_something():
    ...

NOTE: The reason we mock the _timer object instead of the timer functions is
because the timer functions will have already been imported by calling modules
by the time the mock is applied. This means that the mock would not be applied.
"""
import time

import numpy as np

import imprint as ip

logger = ip.getLogger(__name__)


class Timer:
    def __init__(self):
        self.last = np.uint64(0)

    def unique(self):
        now = self.now()
        t = np.uint64(int(now))
        if t <= self.last:
            t = self.last + np.uint64(1)
        self.last = t
        return t

    def now(self):
        return time.time()


class MockTimer:
    def __init__(self):
        self.i = 0

    def unique(self):
        self.i += 1
        return np.uint64(self.i - 1)

    def now(self):
        return self.i


def new_mock_timer():
    return MockTimer()


_timer = Timer()


def unique_timer():
    return _timer.unique()


def simple_timer():
    return _timer.now()
