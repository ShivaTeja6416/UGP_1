from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


CELESTIAL_BODIES = {
    "earth": {"mu": 3.986004418e14, "radius_m": 6378137.0},
    "moon":  {"mu": 4.9048695e12,   "radius_m": 1737400.0},
    "mars":  {"mu": 4.282837e13,    "radius_m": 3389500.0},
    "sun":   {"mu": 1.32712440018e20, "radius_m": 6.957e8},
}


def _validate_positive(**kwargs):
    for name, val in kwargs.items():
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")


# --------------------------------------------------------------------------
# Circular orbit basics
# --------------------------------------------------------------------------

def circular_orbit_velocity(mu: float, r: float) -> float:
    """
    Speed needed to maintain a circular orbit at radius r.
        v = sqrt(mu / r)
    r is measured from the body's CENTER, not altitude above surface.
    """
    _validate_positive(mu=mu, r=r)
    return math.sqrt(mu / r)


def orbital_period(mu: float, a: float) -> float:
    """
    Kepler's third law: period of an orbit with semi-major axis a.
        T = 2*pi*sqrt(a^3 / mu)
    For a circular orbit, a = r.
    """
    _validate_positive(mu=mu, a=a)
    return 2 * math.pi * math.sqrt(a**3 / mu)


def vis_viva(mu: float, r: float, a: float) -> float:
    """
    Vis-viva equation: instantaneous speed at radius r on an orbit
    with semi-major axis a (elliptical, parabolic if a->inf, hyperbolic
    if a<0).
        v = sqrt(mu * (2/r - 1/a))
    """
    _validate_positive(mu=mu, r=r)
    if a == 0:
        raise ValueError("a cannot be 0 (undefined orbit shape).")
    return math.sqrt(mu * (2 / r - 1 / a))


def escape_velocity(mu: float, r: float) -> float:
    """
    Minimum speed to escape a gravity well from radius r (parabolic
    trajectory, a -> infinity).
        v_esc = sqrt(2*mu / r)
    """
    _validate_positive(mu=mu, r=r)
    return math.sqrt(2 * mu / r)


def characteristic_energy_c3(v_infinity_mps: float) -> float:
    """
    C3 = v_infinity^2  [m^2/s^2] -- the standard "how much excess energy
    does this departure trajectory have" metric used in interplanetary
    mission design (JPL uses this constantly).
    """
    if v_infinity_mps < 0:
        raise ValueError("v_infinity_mps must be >= 0.")
    return v_infinity_mps ** 2


# --------------------------------------------------------------------------
# Hohmann transfer
# --------------------------------------------------------------------------

@dataclass
class HohmannResult:
    dv1_mps: float          # burn at the departure orbit
    dv2_mps: float          # burn at arrival to circularize
    total_dv_mps: float
    transfer_time_s: float  # half the transfer ellipse's period


def hohmann_transfer(mu: float, r1: float, r2: float) -> HohmannResult:
    """
    Classic two-burn minimum-energy transfer between two circular,
    coplanar orbits of radii r1 (start) and r2 (target).

    Method (Curtis Ch. 6 / any orbital mechanics text):
        a_transfer = (r1 + r2) / 2
        v1_circular = sqrt(mu/r1)
        v_transfer_at_r1 = vis_viva(mu, r1, a_transfer)
        dv1 = v_transfer_at_r1 - v1_circular

        v2_circular = sqrt(mu/r2)
        v_transfer_at_r2 = vis_viva(mu, r2, a_transfer)
        dv2 = v2_circular - v_transfer_at_r2

        transfer_time = orbital_period(mu, a_transfer) / 2
    """
    _validate_positive(mu=mu, r1=r1, r2=r2)

    a_transfer = (r1 + r2) / 2

    v1_circ = circular_orbit_velocity(mu, r1)
    v_trans_r1 = vis_viva(mu, r1, a_transfer)
    dv1 = v_trans_r1 - v1_circ

    v2_circ = circular_orbit_velocity(mu, r2)
    v_trans_r2 = vis_viva(mu, r2, a_transfer)
    dv2 = v2_circ - v_trans_r2

    t_transfer = orbital_period(mu, a_transfer) / 2

    return HohmannResult(
        dv1_mps=dv1, dv2_mps=dv2,
        total_dv_mps=abs(dv1) + abs(dv2),
        transfer_time_s=t_transfer,
    )


# --------------------------------------------------------------------------
# Plane change
# --------------------------------------------------------------------------

def plane_change_delta_v(v_mps: float, delta_i_deg: float) -> float:
    """
    Delta-v to rotate the orbital plane by delta_i, done as a single
    burn at constant speed v (the standard simple-plane-change formula):
        dv = 2 * v * sin(delta_i / 2)
    Cheapest done where v is smallest (e.g. apoapsis), which is a
    mission-design choice, not something this pure function decides.
    """
    _validate_positive(v_mps=v_mps)
    if delta_i_deg < 0:
        raise ValueError("delta_i_deg must be >= 0.")
    return 2 * v_mps * math.sin(math.radians(delta_i_deg) / 2)


# --------------------------------------------------------------------------
# Generic query dispatcher (same shape as rocket_equation.solve())
# --------------------------------------------------------------------------

_OPS = {
    "circular_orbit_velocity": circular_orbit_velocity,
    "orbital_period": orbital_period,
    "vis_viva": vis_viva,
    "escape_velocity": escape_velocity,
    "characteristic_energy_c3": characteristic_energy_c3,
    "plane_change_delta_v": plane_change_delta_v,
}


def solve(query: dict) -> dict:
    """
    Same calling convention as rocket_equation.solve(), so the future
    router (Phase 1D) can treat every tool identically:

        query = {"operation": "<name>", "inputs": {...}}

    hohmann_transfer is handled separately below since it returns a
    structured result (two burns + transfer time), not a single scalar.
    """
    op = query.get("operation")
    inputs = query.get("inputs", {})

    if op == "hohmann_transfer":
        try:
            result = hohmann_transfer(**inputs)
            return {
                "operation": op, "inputs": inputs,
                "result": {
                    "dv1_mps": result.dv1_mps, "dv2_mps": result.dv2_mps,
                    "total_dv_mps": result.total_dv_mps,
                    "transfer_time_s": result.transfer_time_s,
                },
                "status": "validated",
            }
        except (TypeError, ValueError) as e:
            return {"operation": op, "inputs": inputs, "error": str(e)}

    if op not in _OPS:
        return {"operation": op, "error": f"Unknown operation. Valid: {list(_OPS) + ['hohmann_transfer']}"}
    try:
        result = _OPS[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}
