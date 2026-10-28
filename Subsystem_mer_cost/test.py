
import math
from main import (
    mass_closure_percentage_method, mer_power_law, mer_linear,
    cost_estimate, subsystem_cost_breakdown, point_design_estimate, solve,
)


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected} (tol={tol})")
    assert ok, f"{name} FAILED"


# --- Test 1: mass closure, hand calc
# payload=100kg, fractions: structure=0.2, power=0.2, adcs=0.1 -> sum=0.5
# total = 100 / (1-0.5) = 200 kg
# structure=40, power=40, adcs=20
# IDENTITY CHECK: payload + sum(subsystem masses) must equal total_dry_mass
def test_mass_closure():
    result = mass_closure_percentage_method(
        payload_mass_kg=100,
        subsystem_fractions={"structure": 0.2, "power": 0.2, "adcs": 0.1},
    )
    check("total_dry_mass", result["total_dry_mass_kg"], 200.0, tol=1e-9)
    check("structure_mass", result["subsystem_masses_kg"]["structure"], 40.0, tol=1e-9)
    check("power_mass", result["subsystem_masses_kg"]["power"], 40.0, tol=1e-9)
    check("adcs_mass", result["subsystem_masses_kg"]["adcs"], 20.0, tol=1e-9)

    # conservation identity
    total_check = result["payload_mass_kg"] + sum(result["subsystem_masses_kg"].values())
    check("mass_conservation_identity", total_check, result["total_dry_mass_kg"], tol=1e-9)


# --- Test 2: mer_power_law, hand calc
# a=0.01, b=0.8, x=100 -> mass = 0.01 * 100^0.8
def test_mer_power_law():
    m = mer_power_law(x=100, a=0.01, b=0.8)
    expected = 0.01 * 100 ** 0.8
    check("mer_power_law", m, expected, tol=1e-9)


# --- Test 3: mer_linear, hand calc
# a=2, b=5, x=10 -> mass = 2*10+5 = 25
def test_mer_linear():
    m = mer_linear(x=10, a=2, b=5)
    check("mer_linear", m, 25.0, tol=1e-9)


# --- Test 4: cost_estimate, hand calc
# mass=50kg, cost_per_kg=10000, fixed=5000 -> cost = 50*10000+5000 = 505000
def test_cost_estimate():
    c = cost_estimate(mass_kg=50, cost_per_kg=10000, fixed_cost=5000)
    check("cost_estimate", c, 505000.0, tol=1e-6)


# --- Test 5: subsystem_cost_breakdown, including the "missing rate ->
# skipped, not silently zero" behavior
def test_subsystem_cost_breakdown():
    masses = {"structure": 40, "power": 40, "adcs": 20}
    rates = {"structure": 5000, "power": 8000}  # adcs rate deliberately missing
    result = subsystem_cost_breakdown(masses, rates)
    check("structure_cost", result["per_subsystem_cost"]["structure"], 200000.0, tol=1e-6)
    check("power_cost", result["per_subsystem_cost"]["power"], 320000.0, tol=1e-6)
    check("total_cost", result["total_cost"], 520000.0, tol=1e-6)
    assert result["skipped_subsystems_no_rate_given"] == ["adcs"], \
        "adcs should be flagged as skipped, not silently costed at 0"
    print(f"[PASS] adcs correctly flagged as skipped (no rate given), not silently zeroed")


# --- Test 6: point_design_estimate combined call
def test_point_design_estimate():
    result = point_design_estimate(
        payload_mass_kg=100,
        subsystem_fractions={"structure": 0.2, "power": 0.2, "adcs": 0.1},
        cost_per_kg={"structure": 5000, "power": 8000, "adcs": 6000},
    )
    check("combined_total_dry_mass", result["total_dry_mass_kg"], 200.0, tol=1e-9)
    expected_total_cost = 40 * 5000 + 40 * 8000 + 20 * 6000  # 200000+320000+120000=640000
    check("combined_total_cost", result["total_cost"], expected_total_cost, tol=1e-6)


# --- Test 7: dispatcher
def test_solve_dispatcher():
    r = solve({"operation": "cost_estimate",
               "inputs": {"mass_kg": 50, "cost_per_kg": 10000, "fixed_cost": 5000}})
    assert r["status"] == "validated"
    check("dispatcher_cost", r["result"], 505000.0, tol=1e-6)


# --- Test 8: bad input handling (fractions summing >= 1)
def test_invalid_inputs():
    try:
        mass_closure_percentage_method(100, {"a": 0.6, "b": 0.5})  # sums to 1.1
        assert False, "expected ValueError for fractions summing >= 1"
    except ValueError:
        print("[PASS] fractions summing >= 1 correctly raised ValueError")


if __name__ == "__main__":
    test_mass_closure()
    test_mer_power_law()
    test_mer_linear()
    test_cost_estimate()
    test_subsystem_cost_breakdown()
    test_point_design_estimate()
    test_solve_dispatcher()
    test_invalid_inputs()
    print("\nAll tests passed.")
