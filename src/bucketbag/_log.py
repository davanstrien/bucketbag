"""bucketbag's default log output: ``[bucketbag] message`` lines on stderr while nobody else
has configured logging (the same design as saturate's).

Every module logs to the ``bucketbag`` logger. At import it gets one fallback handler and,
unless the caller has set one, the level INFO, so progress lines and the run report reach a
Job's log without any setup. Once the caller puts a handler on the root logger or on
``bucketbag``, the fallback prints nothing and the caller's handlers get the records. To show
only problems: ``logging.getLogger("bucketbag").setLevel(logging.WARNING)``.
"""

from __future__ import annotations

import logging
import sys

PACKAGE = "bucketbag"


class _Fallback(logging.Handler):
    """Prints to ``sys.stderr`` only while nobody has configured logging.

    ``sys.stderr`` is read at each record: pytest, notebooks and agents replace it after import.
    """

    def __init__(self, package: logging.Logger) -> None:
        super().__init__()
        # Held here, not looked up per record: logging.getLogger takes logging's module lock,
        # and emit runs under this handler's lock, while dictConfig takes them the other way.
        self.package = package
        self.setFormatter(logging.Formatter("[bucketbag] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        theirs = [handler for handler in self.package.handlers if handler is not self]
        if theirs or logging.root.handlers:
            return
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def install() -> None:
    """Attach the fallback handler once."""
    package = logging.getLogger(PACKAGE)
    if any(isinstance(handler, _Fallback) for handler in package.handlers):
        return
    package.addHandler(_Fallback(package))
    if package.level == logging.NOTSET:  # progress lines are INFO; a caller's own level wins
        package.setLevel(logging.INFO)
