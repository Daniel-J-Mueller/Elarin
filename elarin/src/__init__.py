"""Elarin brain modules."""

from __future__ import annotations

__all__ = ["main"]

from .subthalamic_nucleus import SubthalamicNucleus


def main() -> None:
    """Entry point for the demo script."""
    from .brain import main as run

    run()
