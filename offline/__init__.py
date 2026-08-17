"""Offline RL contribution: trajectory logging, corpus collection, dataset, Decision Transformer.

This package is the project's own contribution and sits outside the frozen platform
code.  Nothing here modifies ``envs/``, ``agent/`` or ``experiments/``; the logger
observes an episode through callbacks and the collector reuses
``experiments.envs.make_env`` unchanged.

On-disk format governed by ``docs/CONTRACTS.md`` C6, format version ``"1.0"``.
"""

from __future__ import annotations

__all__ = ["trajectory_logger"]
