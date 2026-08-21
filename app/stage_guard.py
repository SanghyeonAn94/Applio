"""Validate that each training stage produced usable output."""
import logging
import os
from collections.abc import Iterable
from typing import Optional, Union

logger = logging.getLogger(__name__)


class StageProducedNothing(RuntimeError):
    """A stage ran over inputs and wrote no outputs."""


SuffixFilter = Optional[Union[str, Iterable[str]]]


def _normalize_suffixes(suffix: SuffixFilter) -> Optional[tuple[str, ...]]:
    """Return a lowercase tuple suitable for ``str.endswith``."""
    if suffix is None:
        return None
    if isinstance(suffix, str):
        return (suffix.lower(),)
    return tuple(item.lower() for item in suffix)


def count_files(path: str, suffix: SuffixFilter = None) -> int:
    """Count files under a directory, recursively, so nesting cannot hide them."""
    if not os.path.isdir(path):
        return 0
    suffixes = _normalize_suffixes(suffix)
    total = 0
    for _root, _dirs, files in os.walk(path):
        for name in files:
            if suffixes is None or name.lower().endswith(suffixes):
                total += 1
    return total


def require_output(stage: str, in_dir: str, out_dir: str, *,
                   in_suffix: SuffixFilter = None,
                   out_suffix: SuffixFilter = None) -> int:
    """Report what a stage moved and reject an empty result."""
    n_in = count_files(in_dir, in_suffix)
    n_out = count_files(out_dir, out_suffix)
    logger.info("[Stage] %s: %d in (%s) -> %d out (%s)", stage, n_in, in_dir, n_out, out_dir)
    if n_in and not n_out:
        raise StageProducedNothing(
            f"{stage} read {n_in} file(s) from {in_dir} and wrote nothing to {out_dir}"
        )
    return n_out
