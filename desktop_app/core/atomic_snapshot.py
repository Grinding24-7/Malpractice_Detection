"""Lock-free single-writer / multi-reader snapshot slot.

CPython performs object-reference assignment atomically under the GIL.  A
producer can therefore build a fresh, never-mutated snapshot object and swap
one reference, while any number of readers call :meth:`get` without taking a
lock.  Readers that grabbed the previous reference keep a consistent view
because the producer never mutates a published object.

This is deliberately not a general-purpose concurrent container.  It is for
telemetry where dropping an intermediate frame is correct: the UI wants the
latest state, not every state.

Rules for producers:
  * build the complete object first;
  * publish it with :meth:`publish`;
  * never mutate it afterwards (nested lists included).

Rules for readers:
  * treat the returned object as read-only;
  * do not cache it as "the current state" unless comparing by identity.
"""

from __future__ import annotations

import threading
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class AtomicSnapshot(Generic[T]):
    """A lock-free slot holding the latest immutable snapshot.

    ``version`` is published in the same tuple as the value, so a reader can
    detect change without inspecting the payload.  Only one thread may call
    :meth:`publish`; concurrent publishers would make the version ordering
    meaningless (the value itself would still be safe, just last-writer-wins).
    """

    __slots__ = ("_state", "_publish_lock")

    def __init__(self, initial: T) -> None:
        self._state: tuple[int, T] = (0, initial)
        # Only producers serialise.  Readers never touch this lock, which is
        # the entire point: a slow or stalled reader cannot block inference.
        self._publish_lock = threading.Lock()

    def get(self) -> T:
        """Return the latest snapshot by reference, without locking."""
        return self._state[1]

    def state(self) -> tuple[int, T]:
        """Return ``(version, snapshot)`` as one atomically-published tuple."""
        return self._state

    def publish(self, value: T) -> int:
        """Swap in a fully-built snapshot and return its new version."""
        with self._publish_lock:
            version = self._state[0] + 1
            self._state = (version, value)
            return version


__all__ = ["AtomicSnapshot"]
