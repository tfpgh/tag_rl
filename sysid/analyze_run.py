from __future__ import annotations

import json
import sys

from sysid.load_runs import discover_run_dirs, load_run
from sysid.stats import summarize_run


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m sysid.analyze_run <run_dir_or_parent> [more_runs...]")
        raise SystemExit(1)

    runs = [load_run(path) for path in discover_run_dirs(sys.argv[1:])]
    payload = [summarize_run(run) for run in runs]
    if len(payload) == 1:
        print(json.dumps(payload[0], indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
