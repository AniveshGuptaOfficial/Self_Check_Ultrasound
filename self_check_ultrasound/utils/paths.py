"""
Safe path helpers (no torch dependency).
"""

from pathlib import Path
from typing import Union


def resolve_safe_path(root: Union[str, Path], relative_path: Union[str, Path]) -> Path:
    """
    Resolve `relative_path` under `root` and guarantee the result stays inside `root`.

    Uses Path.is_relative_to on fully resolved paths, so a sibling directory that merely
    shares a string prefix (e.g. '/data/root_evil' vs '/data/root') is correctly rejected,
    which a plain str.startswith() check would not do. Symlinks are resolved first.

    Raises ValueError("Path traversal detected: ...") if the path escapes the root.
    """
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relative_path).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"Path traversal detected: {candidate} is outside {root_resolved}")
    return candidate
