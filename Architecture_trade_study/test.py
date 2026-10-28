
from main import (
    enumerate_combinations, apply_compatibility_rules,
    aggregate_criteria, weighted_score, score_combination,
    run_trade_study, solve,
)

PROPULSION = [
    {"name": "mono", "cost": 2, "mass": 5, "reliability": 8},
    {"name": "bi", "cost": 4, "mass": 8, "reliability": 9},
]
POWER = [
    {"name": "solar", "cost": 3, "mass": 6, "reliability": 7},
    {"name": "rtg", "cost": 9, "mass": 4, "reliability": 9},
]
DECISION_VARS = {"propulsion": PROPULSION, "power": POWER}
WEIGHTS = {"cost": -1, "mass": -0.5, "reliability": 2}


def no_mono_rtg(combo):
    return not (combo["propulsion"]["name"] == "mono" and combo["power"]["name"] == "rtg")


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected} (tol={tol})")
    assert ok, f"{name} FAILED"


# --- Test 1: enumeration produces all 4 combinations
def test_enumerate_combinations():
    combos = enumerate_combinations(DECISION_VARS)
    check("num_combinations", len(combos), 4, tol=0)


# --- Test 2: compatibility rule removes exactly 1 (mono+rtg)
def test_apply_compatibility_rules():
    combos = enumerate_combinations(DECISION_VARS)
    valid = apply_compatibility_rules(combos, [no_mono_rtg])
    check("num_valid_after_rule", len(valid), 3, tol=0)
    names = [(c["propulsion"]["name"], c["power"]["name"]) for c in valid]
    assert ("mono", "rtg") not in names, "mono+rtg should have been excluded"
    print(f"[PASS] mono+rtg correctly excluded; remaining: {names}")


# --- Test 3: aggregate_criteria hand calc for mono+solar
def test_aggregate_criteria():
    combo = {"propulsion": PROPULSION[0], "power": POWER[0]}  # mono+solar
    agg = aggregate_criteria(combo)
    check("agg_cost", agg["cost"], 5, tol=1e-9)
    check("agg_mass", agg["mass"], 11, tol=1e-9)
    check("agg_reliability", agg["reliability"], 15, tol=1e-9)


# --- Test 4: weighted_score hand calc for mono+solar -> 19.5
def test_weighted_score():
    combo = {"propulsion": PROPULSION[0], "power": POWER[0]}
    score = score_combination(combo, WEIGHTS)
    check("mono_solar_score", score, 19.5, tol=1e-9)


# --- Test 5: full trade study ranking order
def test_run_trade_study_ranking():
    results = run_trade_study(DECISION_VARS, WEIGHTS, compatibility_rules=[no_mono_rtg])
    check("num_results", len(results), 3, tol=0)

    order = [(r["combo"]["propulsion"]["name"], r["combo"]["power"]["name"]) for r in results]
    expected_order = [("mono", "solar"), ("bi", "solar"), ("bi", "rtg")]
    print(f"[{'PASS' if order == expected_order else 'FAIL'}] ranking order: {order} "
          f"(expected {expected_order})")
    assert order == expected_order

    check("score_1", results[0]["score"], 19.5, tol=1e-9)
    check("score_2", results[1]["score"], 18.0, tol=1e-9)
    check("score_3", results[2]["score"], 17.0, tol=1e-9)


# --- Test 6: dispatcher (plain-data op only)
def test_solve_dispatcher():
    combo = {"propulsion": PROPULSION[0], "power": POWER[0]}
    r = solve({"operation": "score_combination", "inputs": {"combo": combo, "weights": WEIGHTS}})
    assert r["status"] == "validated"
    check("dispatcher_score", r["result"], 19.5, tol=1e-9)


# --- Test 7: bad input handling (empty decision variables)
def test_invalid_inputs():
    try:
        enumerate_combinations({})
        assert False, "expected ValueError for empty decision_variables"
    except ValueError:
        print("[PASS] empty decision_variables correctly raised ValueError")


if __name__ == "__main__":
    test_enumerate_combinations()
    test_apply_compatibility_rules()
    test_aggregate_criteria()
    test_weighted_score()
    test_run_trade_study_ranking()
    test_solve_dispatcher()
    test_invalid_inputs()
    print("\nAll tests passed.")
