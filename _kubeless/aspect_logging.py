#!/usr/bin/env python
from __future__ import print_function

import functools


class CallLoggingAspect():
    """Aspect that adds logging for a method."""

    TEMPLATE = "[  AOP:  {:10} on {:15}] [Calls: {:3}|Ok: {:3}|Failed: {:3}] {}"
    def __init__(self, enter_message, exit_message):
        self._enter_message = enter_message
        self._exit_message = exit_message
        self._enter_call_count = 0
        self._safe_exit_count = 0
        self._failed_exit_count = 0

    def _get_message(self, func, message):
        return self.TEMPLATE.format(
            self.__class__.__name__,
            func.__name__,
            self._enter_call_count,
            self._safe_exit_count,
            self._failed_exit_count,
            message)

    def __call__(self, func):
        """Return the decorated function with the aspect."""
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                print(self._get_message(func, self._enter_message))

                self._enter_call_count += 1
                res = func(*args, **kwargs)
                self._safe_exit_count += 1

                print(self._get_message(func, self._exit_message))

                return res
            except Exception as exc:
                self._failed_exit_count += 1
                raise exc
        return wrapped
