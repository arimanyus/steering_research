"""Eval side of the two-neuron "evil villain" steering claim (see steer/PROTOCOL.md).

No model code lives here: the steering harness emits `outputs.jsonl`, this package scores it.
"""

import json
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_probe_set(probes_dir, name):
    """Rows of `<name>.jsonl`, else of the id manifest `<name>.ids.jsonl` (ids and scorer metadata,
    no prompt text; how withheld sets are tracked), else []."""
    for suffix in (".jsonl", ".ids.jsonl"):
        path = os.path.join(probes_dir, name + suffix)
        if os.path.exists(path):
            return load_jsonl(path)
    return []
