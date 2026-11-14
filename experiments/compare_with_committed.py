"""Compare the files regenerated in a clone with the committed ones.

Procedure (plan.md, item 0.4 and Phase 6):
    git clone <this repository> ../FAI_clean && cd ../FAI_clean
    uv sync --frozen --extra dev
    rm -rf results report/generated report/figures        # run_x1.py and run_cce.py skip cells whose result exists!
    X1_WORKERS=2 X3_WORKERS=2 CCE_WORKERS=1 bash run_all.sh > ../clean_run.log 2>&1
    python experiments/compare_with_committed.py .          # from the clone, or pass the clone's path

Compares every tracked file under results/, report/generated/ and report/figures/ with its version at HEAD of the clone:
  JSON and NPZ  numerically (relative 1e-9, absolute 1e-12), timing fields ignored
  LaTeX         exactly (line endings ignored)
  logs          with elapsed-time tokens and warnings stripped
  PNG and PDF   listed only (matplotlib output is not byte-stable across machines)
"""
import io
import json
import os
import re
import subprocess
import sys

import numpy as np

CLONE = sys.argv[1] if len(sys.argv) > 1 else "."
TIMING_KEYS = {"seconds", "table_seconds", "build_seconds", "lp_seconds", "seconds_per_learner"}


def git(*args, binary=False):
    out = subprocess.run(["git", "-C", CLONE, *args], capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf8", "replace")


def close(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in TIMING_KEYS:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: only in {'committed' if k in a else 'regenerated'}")
            else:
                close(a[k], b[k], f"{path}/{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (abs(a - b) <= 1e-12 + 1e-9 * max(abs(a), abs(b))):
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


def norm_log(text):
    text = re.sub(r"\(\d+(\.\d+)?s\)", "(Ns)", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*s\b", "Ns", text)
    text = re.sub(r"in \d+s", "in Ns", text)
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip() and "Warning" not in ln]


def main():
    files = [f for f in git("ls-files", "results", "report/generated", "report/figures").split("\n") if f]
    summary = {"identical": 0, "differs": 0, "missing": 0, "binary_listed": 0}
    details = []
    for f in files:
        path = os.path.join(CLONE, *f.split("/"))
        try:
            new = open(path, "rb").read()
        except FileNotFoundError:
            summary["missing"] += 1
            details.append(f"MISSING in clone: {f}")
            continue
        old = git("show", f"HEAD:{f}", binary=True)
        ext = f.rsplit(".", 1)[-1]
        diffs = []
        if ext == "json":
            close(json.loads(old), json.loads(new), "", diffs)
        elif ext == "npz":
            zo, zn = np.load(io.BytesIO(old), allow_pickle=True), np.load(io.BytesIO(new), allow_pickle=True)
            for k in sorted(set(zo.files) | set(zn.files)):
                if k in TIMING_KEYS:
                    continue
                if k not in zo.files or k not in zn.files:
                    diffs.append(f"{k}: only in one")
                    continue
                a, b = zo[k], zn[k]
                if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                        diffs.append(f"{k}: arrays differ")
                elif not np.array_equal(a, b):
                    diffs.append(f"{k}: differs")
        elif ext in ("png", "pdf"):
            summary["binary_listed"] += 1
            continue
        elif ext == "log":
            a, b = norm_log(old.decode("utf8", "replace")), norm_log(new.decode("utf8", "replace"))
            if a != b:
                only_old = [x for x in a if x not in set(b)]
                only_new = [x for x in b if x not in set(a)]
                diffs.append(f"{len(only_old)} committed lines absent / {len(only_new)} regenerated lines new; e.g. {only_old[:1]} -> {only_new[:1]}")
        elif old.replace(b"\r\n", b"\n") != new.replace(b"\r\n", b"\n"):
            diffs.append("text differs")
        if diffs:
            summary["differs"] += 1
            details.append(f"DIFFERS {f}:")
            details.extend("    " + d for d in diffs[:6])
            if len(diffs) > 6:
                details.append(f"    ... and {len(diffs) - 6} more")
        else:
            summary["identical"] += 1
    print(json.dumps(summary))
    print("\n".join(details))


if __name__ == "__main__":
    main()
