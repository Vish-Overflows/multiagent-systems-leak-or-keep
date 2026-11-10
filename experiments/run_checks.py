"""Verification checks (Table 1 of the report).  Writes results/checks/checks.json and a log."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.checks import (lemma1_monotonicity, local_gains_complete_n6, no_sharing_at_alpha0,  # noqa: E402
                             prop1_complete_n4, prop2_bar_checks, prop3_leak_formula, worked_example)
from leakgame.config import GameConfig  # noqa: E402
from leakgame.tables import build_table  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "checks")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = {}
    print("C1 worked example ...", flush=True)
    res["C1_worked_example"] = worked_example()
    print(json.dumps(res["C1_worked_example"], indent=1))
    print("C2 Lemma 1 monotonicity ...", flush=True)
    res["C2_lemma1"] = lemma1_monotonicity()
    print(json.dumps(res["C2_lemma1"], indent=1))
    print("C3 Prop 3 leak formula ...", flush=True)
    res["C3_prop3"] = prop3_leak_formula()
    print(json.dumps(res["C3_prop3"], indent=1))
    print("C4 Prop 2 BAR checks (pilot tables) ...", flush=True)
    tables = {e: build_table(GameConfig(env=e), verbose=False) for e in ("E0", "E1", "E2")}
    res["C4_prop2"] = {e: prop2_bar_checks(tables[e]) for e in tables}
    print(json.dumps(res["C4_prop2"], indent=1))
    print("C5 Prop 1 complete network ...", flush=True)
    res["C5_prop1_n4"] = prop1_complete_n4()
    print(json.dumps(res["C5_prop1_n4"], indent=1))
    res["C5_prop1_n6_local"] = {f"alpha={a}": local_gains_complete_n6(alpha=a) for a in (0.2, 1.0)}
    print(json.dumps(res["C5_prop1_n6_local"], indent=1))
    print("C6 nobody shares at alpha = 0 ...", flush=True)
    res["C6_alpha0"] = no_sharing_at_alpha0(tables)
    print(json.dumps(res["C6_alpha0"], indent=1))
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done in {res['seconds']:.1f}s -> {OUT}/checks.json")


if __name__ == "__main__":
    main()
