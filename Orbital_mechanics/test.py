import math
from main import (
    circular_orbit_velocity, orbital_period, vis_viva, escape_velocity,
    characteristic_energy_c3, hohmann_transfer, plane_change_delta_v,
    solve, CELESTIAL_BODIES,
)

MU_EARTH = CELESTIAL_BODIES["earth"]["mu"]
R_EARTH = CELESTIAL_BODIES["earth"]["radius_m"]


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.4f} expected={expected:.4f} (tol={tol})")
    assert ok, f"{name} FAILED"


# --- Test 1: LEO circular velocity, well-known ~7.78 km/s at 200km altitude
def test_leo_circular_velocity():
    r = R_EARTH + 200_000  # 200 km altitude
    v = circular_orbit_velocity(MU_EARTH, r)
    check("leo_circular_velocity", v, 7784.0, tol=5.0)


# --- Test 2: Earth surface escape velocity, textbook value ~11.186 km/s
def test_earth_escape_velocity():
    v = escape_velocity(MU_EARTH, R_EARTH)
    # 11186 m/s is the commonly-quoted value using Earth's MEAN radius
    # (6371 km); we use the equatorial radius (6378.137 km) as our
    # constant, which correctly gives a very slightly lower number.
    # Widening tolerance to reflect that known, legitimate difference.
    check("earth_escape_velocity", v, 11186.0, tol=10.0)


# --- Test 3: ISS-altitude orbital period, well-known ~92.7 minutes (~5561 s)
def test_iss_period():
    r = R_EARTH + 400_000  # ~ISS altitude
    T = orbital_period(MU_EARTH, r)
    check("iss_orbital_period_s", T, 92.7 * 60, tol=30.0)


# --- Test 4: vis-viva at r = a should equal circular velocity (sanity identity)
def test_vis_viva_circular_case():
    r = R_EARTH + 500_000
    v_circ = circular_orbit_velocity(MU_EARTH, r)
    v_vis = vis_viva(MU_EARTH, r, a=r)  # a == r means it IS a circular orbit
    check("vis_viva_matches_circular", v_vis, v_circ, tol=0.01)


# --- Test 5: classic LEO(200km) -> GEO Hohmann transfer.
# Standard textbook answer: dv1 ~= 2457 m/s, dv2 ~= 1478 m/s,
# total ~= 3935 m/s, transfer time ~= 5.26 hours (~18936 s)
def test_hohmann_leo_to_geo():
    r1 = R_EARTH + 200_000
    r2 = 42_164_000  # GEO radius from Earth's center
    result = hohmann_transfer(MU_EARTH, r1, r2)
    check("hohmann_dv1", result.dv1_mps, 2457.0, tol=15.0)
    check("hohmann_dv2", result.dv2_mps, 1478.0, tol=15.0)
    check("hohmann_total_dv", result.total_dv_mps, 3935.0, tol=20.0)
    check("hohmann_transfer_time_hours", result.transfer_time_s / 3600, 5.26, tol=0.05)


# --- Test 6: plane change delta-v, hand calc
# v=1000 m/s, delta_i=30deg -> dv = 2*1000*sin(15deg) = 517.638 m/s
def test_plane_change():
    dv = plane_change_delta_v(v_mps=1000, delta_i_deg=30)
    expected = 2 * 1000 * math.sin(math.radians(15))
    check("plane_change_delta_v", dv, expected, tol=1e-6)


# --- Test 7: C3, trivial hand calc: v_inf=3000 m/s -> C3=9,000,000 m^2/s^2
def test_c3():
    c3 = characteristic_energy_c3(3000)
    check("c3", c3, 9_000_000, tol=1e-6)


# --- Test 8: dispatcher (router-shaped call), including the structured
# hohmann_transfer result
def test_solve_dispatcher():
    r = solve({"operation": "circular_orbit_velocity",
               "inputs": {"mu": MU_EARTH, "r": R_EARTH + 200_000}})
    assert r["status"] == "validated"
    check("solve_circular_velocity", r["result"], 7784.0, tol=5.0)

    r2 = solve({"operation": "hohmann_transfer",
                "inputs": {"mu": MU_EARTH, "r1": R_EARTH + 200_000, "r2": 42_164_000}})
    assert r2["status"] == "validated"
    check("solve_hohmann_total_dv", r2["result"]["total_dv_mps"], 3935.0, tol=20.0)


# --- Test 9: bad input handling
def test_invalid_inputs():
    try:
        circular_orbit_velocity(mu=-1, r=7000000)
        assert False, "expected ValueError for negative mu"
    except ValueError:
        print("[PASS] negative mu correctly raised ValueError")


if __name__ == "__main__":
    test_leo_circular_velocity()
    test_earth_escape_velocity()
    test_iss_period()
    test_vis_viva_circular_case()
    test_hohmann_leo_to_geo()
    test_plane_change()
    test_c3()
    test_solve_dispatcher()
    test_invalid_inputs()
    print("\nAll tests passed.")
