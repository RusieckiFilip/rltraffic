"""Behaviour policies for offline-corpus collection (P2.5+).

Each policy is registered into :data:`offline.collect.POLICIES` and returns a
callable ``act(info) -> np.ndarray`` of ``int64`` actions, one per intersection,
ordered by ``[ix.id for ix in env.intersections]``.
"""

from __future__ import annotations
