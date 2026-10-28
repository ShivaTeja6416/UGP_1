
from main import (
    impact_energy, required_stroke_length, average_impact_force,
    stroke_from_force_and_energy, deceleration_from_stroke,
    attenuator_mass, design_attenuator, solve, G0,
)


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.4f} expected={expected:.4f} (tol={tol})")
    assert ok, f"{name} FAILED"


# --- Test 1: basic impact energy
# mass=500kg, v=5 m/s -> KE = 0.5*500*25 = 6250 J
def test_impact_energy():
    ke = impact_energy(500, 5)
    check("impact_energy", ke, 6250.0, tol=1e-6)


# --- Test 2: required stroke length
# v=5 m/s, allowable_decel=10g -> a=98.0665 m/s^2, s = 25/196.133 = 0.12747 m
def test_required_stroke_length():
    s = required_stroke_length(velocity_mps=5, allowable_decel_g=10)
    expected = 25 / (2 * 10 * G0)
    check("required_stroke_length", s, expected, tol=1e-6)


# --- Test 3: average impact force
# mass=500kg, decel=10g -> F = 500 * 10 * 9.80665 = 49033.25 N
def test_average_impact_force():
    f = average_impact_force(mass_kg=500, allowable_decel_g=10)
    check("average_impact_force", f, 500 * 10 * G0, tol=1e-6)


# --- Test 4: ENERGY CONSERVATION IDENTITY --------------------------------
# This is the important cross-check: force * stroke should equal the
# original kinetic energy, since avg_force = m*a and stroke = v^2/(2a),
# so force*stroke = m*a*v^2/(2a) = 0.5*m*v^2 = KE exactly.
# If this fails, something is wrong with the underlying physics, not
# just a numerical tolerance issue.
def test_energy_conservation_identity():
    mass, v, decel_g = 500, 5, 10
    ke = impact_energy(mass, v)
    s = required_stroke_length(v, decel_g)
    f = average_impact_force(mass, decel_g)
    work_done = f * s
    check("energy_conservation_identity", work_done, ke, tol=1e-6)


# --- Test 5: inverse relationship (deceleration_from_stroke undoes
# required_stroke_length)
def test_stroke_deceleration_roundtrip():
    v, decel_g = 6, 8
    s = required_stroke_length(v, decel_g)
    decel_back = deceleration_from_stroke(v, s)
    check("stroke_deceleration_roundtrip", decel_back, decel_g, tol=1e-6)


# --- Test 6: attenuator mass, simple hand calc
# energy=6250 J, SEA=250000 J/kg -> mass = 6250/250000 = 0.025 kg
def test_attenuator_mass():
    m = attenuator_mass(energy_to_absorb_J=6250, sea_j_per_kg=250000)
    check("attenuator_mass", m, 0.025, tol=1e-9)


# --- Test 7: combined design_attenuator() call, cross-checked against
# the individual functions above (should match exactly)
def test_design_attenuator_combined():
    result = design_attenuator(mass_kg=500, velocity_mps=5,
                                allowable_decel_g=10, sea_j_per_kg=250000)
    check("combined_impact_energy", result.impact_energy_J, 6250.0, tol=1e-6)
    check("combined_stroke", result.stroke_length_m, 25 / (2 * 10 * G0), tol=1e-6)
    check("combined_force", result.avg_force_N, 500 * 10 * G0, tol=1e-6)
    check("combined_attenuator_mass", result.attenuator_mass_kg, 6250 / 250000, tol=1e-6)


# --- Test 8: dispatcher
def test_solve_dispatcher():
    r = solve({"operation": "design_attenuator",
               "inputs": {"mass_kg": 500, "velocity_mps": 5,
                          "allowable_decel_g": 10, "sea_j_per_kg": 250000}})
    assert r["status"] == "validated"
    check("dispatcher_impact_energy", r["result"]["impact_energy_J"], 6250.0, tol=1e-6)


# --- Test 9: bad input handling
def test_invalid_inputs():
    try:
        impact_energy(mass_kg=-500, velocity_mps=5)
        assert False, "expected ValueError for negative mass"
    except ValueError:
        print("[PASS] negative mass correctly raised ValueError")


if __name__ == "__main__":
    test_impact_energy()
    test_required_stroke_length()
    test_average_impact_force()
    test_energy_conservation_identity()
    test_stroke_deceleration_roundtrip()
    test_attenuator_mass()
    test_design_attenuator_combined()
    test_solve_dispatcher()
    test_invalid_inputs()
    print("\nAll tests passed.")
