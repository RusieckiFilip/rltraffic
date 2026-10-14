"""Model windows of the implementer session, from its transcript's metadata only (``type``, ``message.model``,
``message.id``, ``timestamp``): per model, the responses (unique message ids) and the transcript entries, and each
contiguous run of one model with its first and last timestamp.  Reads nothing else.

    python -I docs/returns/P5.2c_evidence/g5fix_model_windows.py <transcript.jsonl> [cut-off ISO timestamp, Z]
"""
from __future__ import annotations

import json
import sys

path = sys.argv[1]
cutoff = sys.argv[2] if len(sys.argv) > 2 else None
entries = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        message = record.get("message")
        if record.get("type") != "assistant" or not isinstance(message, dict):
            continue
        model, stamp = message.get("model"), record.get("timestamp")
        if not (isinstance(model, str) and model.startswith("claude") and stamp):
            continue
        if cutoff and stamp > cutoff:
            continue
        entries.append((stamp, model, message.get("id")))
entries.sort()
totals: dict[str, dict[str, object]] = {}
for stamp, model, message_id in entries:
    block = totals.setdefault(model, {"entries": 0, "ids": set()})
    block["entries"] += 1
    block["ids"].add(message_id)
print(f"# transcript {path}; assistant entries up to {cutoff or 'the end'}: {len(entries)}")
for model, block in sorted(totals.items()):
    print(f"{model}: {len(block['ids'])} responses (unique message ids), {block['entries']} transcript entries")
runs = []
for stamp, model, _ in entries:
    if runs and runs[-1][0] == model:
        runs[-1][2] = stamp
    else:
        runs.append([model, stamp, stamp])
print("contiguous runs of one model:")
for model, first, last in runs:
    print(f"  {model}: {first} .. {last}")
