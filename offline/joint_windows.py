"""Group per-intersection context windows into JOINT windows -- one decision instant, all nodes.

Index format version: ``joint-window-index/1.0``.

WHY THIS EXISTS
---------------
``offline/dataset.py`` yields one item per ``(episode, intersection, t)``: that is the right unit
for a model whose intersections do not see each other, which is every model this project has
trained so far.  A spatial mixing layer needs the other unit -- **all N intersections at the same
decision instant** -- so that node ``i``'s tokens can attend to node ``j``'s.

WHAT IS AND IS NOT RE-DERIVED HERE
-----------------------------------
**Nothing about a window is redefined.**  The slicing, the left padding, the ``PAD_ACTION``
convention and the returns-to-go all stay in ``offline/dataset.py``, and the tensors come from
``offline.dt_gate.stack_dataset``, which builds them by iterating the dataset's own
``__getitem__``.  This module contributes exactly one thing: a table saying **which flat rows
belong to the same decision instant, and in which node order**.

🚨 THE GROUPING IS READ FROM ``item_meta``, NEVER FROM INDEX ARITHMETIC
-----------------------------------------------------------------------
The dataset happens to lay items out as ``(episode, ix_index, t)``, so
``episode_base + ix_index * T + t`` would work today.  It is not used, and the choice is
deliberate: that formula is a **restatement of an internal layout**, and it would keep producing
plausible groups after any change to selection, subsampling or episode ordering.  Every group here
comes from :meth:`TrajectoryWindowDataset.item_meta`, which is the loader's own answer to "what is
this row?".  Same rule as ``PROJECT_PLAN`` section 7's 2026-08-16 pairing-key finding: the key that
pairs two things is proved from the data, never assumed from a layout.

COMPLETENESS IS REQUIRED, NOT REPAIRED
---------------------------------------
A decision instant missing an intersection is **refused**, never padded and never dropped.  C6
forbids padding across intersections, and silently dropping an instant would make the joint corpus
a biased subsample of the per-intersection one -- a difference no downstream number would reveal.

MEMORY
------
``stack_joint`` returns the **flat** tensors plus an ``(M, N)`` ``member_index`` into them, rather
than a materialised ``(M, N, K, D)`` tensor.  The two hold the same number of windows
(``M * N == P``), so the joint form would not cost more -- but gathering per batch keeps one copy
instead of two.  On ``cf_grid4x4__mappo1000`` the flat state tensor is **3.43 GiB** (measured), so
the second copy is the difference between fitting and not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from offline.dataset import TrajectoryWindowDataset

__all__ = [
    "JOINT_INDEX_FORMAT_VERSION",
    "JointWindowIndex",
    "build_joint_index",
    "stack_joint",
]

JOINT_INDEX_FORMAT_VERSION = "joint-window-index/1.0"


@dataclass(frozen=True)
class JointWindowIndex:
    """Which dataset items form each joint decision instant, in a declared node order.

    ``member_index[m, n]`` is the **dataset item index** of node ``node_ids[n]`` at joint window
    ``m``.  ``episode_index[m]`` and ``t[m]`` identify the instant.  The node order is the one the
    caller declared, and it is the same order the adjacency matrix is indexed by -- those two must
    agree or the mixing layer masks the wrong rows.
    """

    node_ids: tuple[str, ...]
    member_index: np.ndarray
    episode_index: np.ndarray
    t: np.ndarray
    state_dim: int
    n_actions: int

    @property
    def n_windows(self) -> int:
        """Number of joint decision instants."""
        return int(self.member_index.shape[0])

    @property
    def n_nodes(self) -> int:
        """Number of intersections per instant."""
        return len(self.node_ids)

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready summary; the index arrays themselves are not serialised."""
        return {
            "format_version": JOINT_INDEX_FORMAT_VERSION,
            "n_windows": self.n_windows,
            "n_nodes": self.n_nodes,
            "node_ids": list(self.node_ids),
            "state_dim": int(self.state_dim),
            "n_actions": int(self.n_actions),
            "grouping": "by (episode_index, t) read from TrajectoryWindowDataset.item_meta",
        }


def build_joint_index(
    dataset: TrajectoryWindowDataset, node_ids: Sequence[str]
) -> JointWindowIndex:
    """Group *dataset*'s items by ``(episode_index, t)``, ordered by *node_ids*.

    Refuses a dataset spanning more than one ``(state_dim, n_actions)`` group, any instant that
    does not carry exactly *node_ids*, and any node id absent from the corpus.  All validation
    runs before the index array is allocated, so a refusal builds nothing.
    """
    order = [str(node) for node in node_ids]
    if not order:
        raise ValueError(
            "node_ids is empty: the node order must come from the data and must be the same "
            "order the adjacency matrix is indexed by"
        )
    duplicates = sorted({node for node in order if order.count(node) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate node ids in the requested order: {duplicates}; each column of the joint "
            "window is one intersection and a repeat would feed the mixing layer the same node twice"
        )

    groups = dataset.groups
    if len(groups) != 1:
        raise ValueError(
            f"this dataset spans {len(groups)} (state_dim, n_actions) groups {sorted(groups)}; "
            "C6 forbids padding across intersections, so a joint window cannot straddle them"
        )
    (state_dim, n_actions), _ = next(iter(groups.items()))

    # The grouping comes from the loader's own provenance, never from index arithmetic.
    members: dict[tuple[int, int], dict[str, int]] = {}
    seen: set[str] = set()
    for item in range(len(dataset)):
        meta = dataset.item_meta(item)
        seen.add(meta.ix_id)
        members.setdefault((int(meta.episode_index), int(meta.t)), {})[meta.ix_id] = item

    wanted = set(order)
    unknown = sorted(wanted - seen)
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested node id(s) are never present in the corpus: {unknown[:8]}; "
            f"the corpus carries {len(seen)} intersections {sorted(seen)[:8]}"
        )
    extra = sorted(seen - wanted)
    if extra:
        raise ValueError(
            f"the corpus carries intersections the node order omits: {extra[:8]} "
            f"({len(extra)} of {len(seen)}). A joint window must span every controlled "
            "intersection; dropping one would make this a biased subsample of the "
            "per-intersection corpus and no downstream number would show it"
        )

    incomplete = [
        (key, sorted(wanted - set(present)))
        for key, present in members.items()
        if len(present) != len(order)
    ]
    if incomplete:
        (episode, step), missing = incomplete[0]
        raise ValueError(
            f"{len(incomplete)} decision instant(s) do not carry every intersection; the first is "
            f"episode {episode} step {step}, missing {missing[:8]}. Such an instant is refused "
            "rather than padded (C6) or dropped (it would bias the corpus)"
        )

    keys = sorted(members)
    member_index = np.array(
        [[members[key][node] for node in order] for key in keys], dtype=np.int64
    )
    return JointWindowIndex(
        node_ids=tuple(order),
        member_index=member_index,
        episode_index=np.array([key[0] for key in keys], dtype=np.int64),
        t=np.array([key[1] for key in keys], dtype=np.int64),
        state_dim=int(state_dim),
        n_actions=int(n_actions),
    )


def stack_joint(
    dataset: TrajectoryWindowDataset, index: JointWindowIndex
) -> dict[str, torch.Tensor]:
    """The flat window tensors plus ``member_index`` remapped into flat-row space.

    Row ``r`` of every returned tensor is ``dataset[item_index[r]]``; ``member_index[m, n]`` is the
    **flat row** of node ``n`` at instant ``m``, so a training batch is
    ``tensor[member_index[rows]]`` with shape ``(B, N, K, ...)``.

    The tensors come from ``offline.dt_gate.stack_dataset``, which builds them from the dataset's
    own ``__getitem__``; this function only re-keys them.
    """
    from offline.dt_gate import stack_dataset

    stacked = stack_dataset(dataset, (index.state_dim, index.n_actions))
    item_index = stacked["item_index"].numpy()

    # dataset item index -> flat row.  Built rather than assumed: stack_dataset stacks one group's
    # indices in dataset order, which is the identity today and is not promised.
    inverse = np.full(len(dataset), -1, dtype=np.int64)
    inverse[item_index] = np.arange(item_index.shape[0], dtype=np.int64)

    flat_members = inverse[index.member_index]
    if int(flat_members.min()) < 0:
        missing = int((flat_members < 0).sum())
        raise ValueError(
            f"{missing} joint members are absent from the stacked group; the index and the "
            "tensors were built from different datasets"
        )
    out = dict(stacked)
    out["member_index"] = torch.from_numpy(flat_members)
    return out
