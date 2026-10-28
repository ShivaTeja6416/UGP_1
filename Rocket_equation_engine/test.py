import math
from main import (
    delta_v, final_mass, initial_mass, required_isp,
    propellant_mass, mass_ratio, propellant_mass_fraction,
    mass_flow_rate, burn_time, thrust_from_mdot,
    Stage, multi_stage_delta_v, solve, G0,
)

TOL = 1e-2  # absolute tolerance in the relevant units (m/s or kg)


def check(name, got, expected, tol=TOL):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.4f} expected={expected:.4f}")
    assert ok, f"{name} FAILED: got {got}, expected {expected}"


# --- Test 1: basic single-burn delta-v --------------------------------
# m0=1000 kg, mf=400 kg, Isp=300 s
# dv = 300 * 9.80665 * ln(1000/400) = 2941.995 * 0.9162907 = 2695.62 m/s
def test_delta_v_basic():
    dv = delta_v(isp_s=300, m0_kg=1000, mf_kg=400)
    check("delta_v_basic", dv, 300 * G0 * math.log(1000 / 400))


# --- Test 2: solve for final mass given a target delta-v --------------
# m0=500 kg, Isp=311 s, target dv=3000 m/s
# mf = 500 * exp(-3000 / (311*9.80665))
def test_final_mass():
    mf = final_mass(isp_s=311, m0_kg=500, delta_v_mps=3000)
    expected = 500 * math.exp(-3000 / (311 * G0))
    check("final_mass", mf, expected)


# --- Test 3: round trip -- final_mass then recover the same delta-v ---
def test_round_trip():
    m0, isp, dv_in = 1200, 250, 1800
    mf = final_mass(isp, m0, dv_in)
    dv_out = delta_v(isp, m0, mf)
    check("round_trip_delta_v", dv_out, dv_in, tol=1e-6)


# --- Test 4: multi-stage delta-v (2 stages) ----------------------------
# Stage 1: Isp=280s, m0=2000kg, mf=800kg
# Stage 2: Isp=320s, m0=750kg,  mf=300kg
def test_multi_stage():
    stages = [
        Stage(isp_s=280, m0_kg=2000, mf_kg=800),
        Stage(isp_s=320, m0_kg=750, mf_kg=300),
    ]
    total = multi_stage_delta_v(stages)
    dv1 = 280 * G0 * math.log(2000 / 800)
    dv2 = 320 * G0 * math.log(750 / 300)
    check("multi_stage_delta_v", total, dv1 + dv2)


# --- Test 5: mass flow rate / burn time consistency --------------------
# Thrust=450 N, Isp=210s -> mdot; then burn_time(prop_mass, mdot) should
# recover a sensible number.
def test_mdot_and_burn_time():
    mdot = mass_flow_rate(thrust_N=450, isp_s=210)
    expected_mdot = 450 / (210 * G0)
    check("mass_flow_rate", mdot, expected_mdot, tol=1e-6)

    t = burn_time(propellant_mass_kg=50, mdot_kgps=mdot)
    expected_t = 50 / expected_mdot
    check("burn_time", t, expected_t, tol=1e-3)


# --- Test 6: required Isp to hit a target dv with a given mass ratio ---
def test_required_isp():
    m0, mf, dv_target = 900, 300, 2500
    isp = required_isp(m0, mf, dv_target)
    # plug back in and confirm it reproduces the target dv
    dv_check = delta_v(isp, m0, mf)
    check("required_isp_roundtrip", dv_check, dv_target, tol=1e-6)


# --- Test 7: generic solve() dispatcher (router-shaped call) -----------
def test_solve_dispatcher():
    result = solve({
        "operation": "delta_v",
        "inputs": {"isp_s": 300, "m0_kg": 1000, "mf_kg": 400}
    })
    assert result["status"] == "validated"
    check("solve_dispatcher_delta_v", result["result"], 300 * G0 * math.log(1000 / 400))


# --- Test 8: bad input handling (mf > m0 should raise, not silently wrong) ---
def test_invalid_mass_order():
    try:
        delta_v(isp_s=300, m0_kg=400, mf_kg=1000)
        assert False, "expected ValueError for mf_kg > m0_kg"
    except ValueError:
        print("[PASS] invalid_mass_order correctly raised ValueError")


if __name__ == "__main__":
    test_delta_v_basic()
    test_final_mass()
    test_round_trip()
    test_multi_stage()
    test_mdot_and_burn_time()
    test_required_isp()
    test_solve_dispatcher()
    test_invalid_mass_order()
    print("\nAll tests passed.")
