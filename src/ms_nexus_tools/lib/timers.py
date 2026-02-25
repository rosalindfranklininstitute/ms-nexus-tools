from typing import Callable, Any
from contextlib import contextmanager, AbstractContextManager
import time
import json
import threading
from collections.abc import Iterable
from pathlib import Path

from icecream import ic


@contextmanager
def time_this(name: str):
    # Code to acquire resource, e.g.:
    now = time.monotonic()
    print(f"{name} began.")
    try:
        yield
    finally:
        # Code to release resource, e.g.:
        print(f"{name} completed in {time.monotonic() - now}s")


class Timer(AbstractContextManager):
    def __init__(
        self,
        name: str,
        interval: float = 120,
        total: int = -1,
        skip_percent: float = -1,
        report_callback: Callable[[], tuple[int, int]] | None = None,
    ):
        self.name = name
        self.interval = interval
        self.total = total
        self.skip_percent = skip_percent
        self.report_callback = report_callback
        assert report_callback is None or total == -1
        self._interval = interval
        self._timer = threading.Timer(interval=interval, function=self.report, args=[])

        self._start: float = -1
        self._last_report_time: float = -1
        self._last_report_percent: float = -skip_percent if skip_percent >= 0 else -1
        self._last_count: int = -1

    def __enter__(self):
        self._start = time.monotonic()
        self._last_report_time = self._start
        self._start_timer()
        if self.report_callback is not None:
            _, total = self.report_callback()
        else:
            total = self.total

        if total > 0:
            print(f"{self.name} began: {total}")
        else:
            print(f"{self.name} began")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._timer.cancel()
        now = time.monotonic()
        print(f"{self.name} completed in {now - self._start:.2f}s")

    def total_time(self):
        return time.monotonic() - self._start

    def _start_timer(self):
        self._timer.cancel()
        self._timer = threading.Timer(
            interval=self.interval,
            function=self._print,
            args=[],
        )
        self._timer.start()

    def _print(self):
        now = time.monotonic()

        if self.report_callback is not None:
            self._last_count, self.total = self.report_callback()
            print_perc = True
        else:
            print_perc = self.total > 0 and self._last_count >= 0
        print(
            f"{self.name}: since last report: {now - self._last_report_time:.2f}s total: {now - self._start:.2f}s",
            end="" if print_perc else None,
        )
        if print_perc:
            percent = (self._last_count + 1) / self.total * 100
            print(f" {self._last_count + 1:d}/{self.total:d}: {percent:.0f}%")
            self._last_report_percent = percent
        self._last_report_time = now
        self._start_timer()

    def report(self, count: int = -1) -> None:
        assert count == -1 or self.report_callback is None
        skip = False
        if count >= 0:
            self._last_count = count
        if self.total > 0 and self._last_count >= 0:
            percent = (self._last_count + 1) / self.total * 100
            next_report_percent = self._last_report_percent + self.skip_percent
            skip = percent < (next_report_percent)

        if skip:
            return

        self._print()


class JSONTimer(AbstractContextManager):
    def __init__(self, filename: Path, keys: Iterable[str]):
        self.filename = filename
        self.keys = keys

        self._start: float = -1
        self._data: dict[str, Any] = {}

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.monotonic()
        duration = end - self._start
        timings = {}
        if self.filename.exists():
            with open(self.filename, "r") as fd:
                timings = json.load(fd)
        new_data = timings
        for key in self.keys:
            if key not in new_data:
                new_data[key] = {}
            new_data = new_data[key]
        new_data["duratioin"] = duration
        new_data.update(self._data)
        with open(self.filename, "w") as fd:
            json.dump(timings, fd, indent=2)

    def add_user_data(self, /, **kwargs):
        self._data.update(kwargs)
