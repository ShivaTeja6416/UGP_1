from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List

G0 = 9.80665  # m/s^2, standard gravity


# --------------------------------------------------------------------------
# Core Tsiolkovsky rocket equation and its inverses
# --------------------------------------------------------------------------

def delta_v(isp_s: float, m0_kg: float, mf_kg: float, g0: float = G0) -> float:

    _validate_positive(isp_s=isp_s, m0_kg=m0_kg, mf_kg=mf_kg)
    if mf_kg > m0_kg:
        raise ValueError("mf_kg cannot exceed m0_kg (final mass > initial mass).")
    return isp_s * g0 * math.log(m0_kg / mf_kg)


def final_mass(isp_s: float, m0_kg: float, delta_v_mps: float, g0: float = G0) -> float:
    """Solve the rocket equation for final mass, given a required delta-v."""
    _validate_positive(isp_s=isp_s, m0_kg=m0_kg)
    if delta_v_mps < 0:
        raise ValueError("delta_v_mps must be >= 0.")
    return m0_kg * math.exp(-delta_v_mps / (isp_s * g0))


def initial_mass(isp_s: float, mf_kg: float, delta_v_mps: float, g0: float = G0) -> float:
    """Solve the rocket equation for required initial (wet) mass."""
    _validate_positive(isp_s=isp_s, mf_kg=mf_kg)
    if delta_v_mps < 0:
        raise ValueError("delta_v_mps must be >= 0.")
    return mf_kg * math.exp(delta_v_mps / (isp_s * g0))


def required_isp(m0_kg: float, mf_kg: float, delta_v_mps: float, g0: float = G0) -> float:
    """Solve the rocket equation for the Isp needed to hit a target delta-v."""
    _validate_positive(m0_kg=m0_kg, mf_kg=mf_kg)
    if mf_kg > m0_kg:
        raise ValueError("mf_kg cannot exceed m0_kg.")
    if delta_v_mps <= 0:
        raise ValueError("delta_v_mps must be > 0 to solve for Isp.")
    return delta_v_mps / (g0 * math.log(m0_kg / mf_kg))


# --------------------------------------------------------------------------
# Mass bookkeeping
# --------------------------------------------------------------------------

def propellant_mass(m0_kg: float, mf_kg: float) -> float:
    """Propellant consumed = m0 - mf."""
    _validate_positive(m0_kg=m0_kg, mf_kg=mf_kg)
    if mf_kg > m0_kg:
        raise ValueError("mf_kg cannot exceed m0_kg.")
    return m0_kg - mf_kg


def mass_ratio(m0_kg: float, mf_kg: float) -> float:
    """m0 / mf -- dimensionless, always >= 1 for a valid burn."""
    _validate_positive(m0_kg=m0_kg, mf_kg=mf_kg)
    return m0_kg / mf_kg


def propellant_mass_fraction(m0_kg: float, mf_kg: float) -> float:
    """(m0 - mf) / m0 -- the classic "how much of the rocket is fuel" number."""
    _validate_positive(m0_kg=m0_kg, mf_kg=mf_kg)
    if mf_kg > m0_kg:
        raise ValueError("mf_kg cannot exceed m0_kg.")
    return (m0_kg - mf_kg) / m0_kg


# --------------------------------------------------------------------------
# Thrust / mass-flow / burn-time relationships
# --------------------------------------------------------------------------

def mass_flow_rate(thrust_N: float, isp_s: float, g0: float = G0) -> float:
    """mdot = F / (Isp * g0)  [kg/s]"""
    _validate_positive(thrust_N=thrust_N, isp_s=isp_s)
    return thrust_N / (isp_s * g0)


def burn_time(propellant_mass_kg: float, mdot_kgps: float) -> float:
    """t_burn = m_prop / mdot  [s]"""
    _validate_positive(propellant_mass_kg=propellant_mass_kg, mdot_kgps=mdot_kgps)
    return propellant_mass_kg / mdot_kgps


def thrust_from_mdot(mdot_kgps: float, isp_s: float, g0: float = G0) -> float:
    """Inverse of mass_flow_rate: F = mdot * Isp * g0  [N]"""
    _validate_positive(mdot_kgps=mdot_kgps, isp_s=isp_s)
    return mdot_kgps * isp_s * g0


# --------------------------------------------------------------------------
# Multi-stage delta-v
# --------------------------------------------------------------------------

@dataclass
class Stage:
    isp_s: float
    m0_kg: float   # mass at ignition of THIS stage (includes all mass above it)
    mf_kg: float   # mass at burnout of THIS stage, before jettison


def multi_stage_delta_v(stages: Iterable[Stage]) -> float:
    """
    Total delta-v across an arbitrary number of stages, fired in sequence.
    Each stage's dv is computed independently and summed -- this is the
    standard "ideal staging" treatment (Curtis Ch. 11 / SEH).
    """
    total = 0.0
    for i, s in enumerate(stages):
        total += delta_v(s.isp_s, s.m0_kg, s.mf_kg)
    return total


def multi_stage_breakdown(stages: Iterable[Stage]) -> List[dict]:
    """Same as multi_stage_delta_v but returns the per-stage numbers too,
    useful for showing your work / debugging a design."""
    out = []
    for i, s in enumerate(stages):
        dv = delta_v(s.isp_s, s.m0_kg, s.mf_kg)
        out.append({
            "stage_index": i + 1,
            "isp_s": s.isp_s,
            "m0_kg": s.m0_kg,
            "mf_kg": s.mf_kg,
            "delta_v_mps": dv,
        })
    return out


# --------------------------------------------------------------------------
# OPTIONAL: finite-burn gravity-loss correction (approximation, labeled as such)
# --------------------------------------------------------------------------

def delta_v_with_gravity_loss(
    ideal_delta_v_mps: float,
    burn_time_s: float,
    g_local_mps2: float = 9.81,
    avg_pitch_deg: float = 90.0,
) -> float:
    """
    APPROXIMATION ONLY -- not a trajectory simulation.

    Rough gravity-loss subtraction for a finite (non-impulsive) burn:
        dv_actual ~= dv_ideal - g_local * t_burn * sin(pitch)

    avg_pitch_deg = 90 means straight up (worst case, max loss);
    0 means horizontal (no gravity loss). Use this only for order-of-
    magnitude sanity checks, and say so in any output that uses it.
    """
    if burn_time_s < 0:
        raise ValueError("burn_time_s must be >= 0.")
    loss = g_local_mps2 * burn_time_s * math.sin(math.radians(avg_pitch_deg))
    return ideal_delta_v_mps - loss


# --------------------------------------------------------------------------
# Generic query dispatcher (this is the shape the Phase 1D router will call)
# --------------------------------------------------------------------------

_OPS = {
    "delta_v": delta_v,
    "final_mass": final_mass,
    "initial_mass": initial_mass,
    "required_isp": required_isp,
    "propellant_mass": propellant_mass,
    "mass_ratio": mass_ratio,
    "propellant_mass_fraction": propellant_mass_fraction,
    "mass_flow_rate": mass_flow_rate,
    "burn_time": burn_time,
    "thrust_from_mdot": thrust_from_mdot,
}


def solve(query: dict) -> dict:
    """
    Generic entry point matching the future router's calling convention.

    query = {
        "operation": "<one of _OPS keys>",
        "inputs": {<kwargs matching that function's signature>}
    }

    Returns {"operation": ..., "inputs": ..., "result": ..., "status": "validated"}
    or {"operation": ..., "error": "..."} on bad input.

    (Multi-stage and gravity-loss ops are handled separately since they
    take structured inputs, not flat kwargs -- see multi_stage_delta_v
    and delta_v_with_gravity_loss above.)
    """
    op = query.get("operation")
    inputs = query.get("inputs", {})
    if op not in _OPS:
        return {"operation": op, "error": f"Unknown operation. Valid: {list(_OPS)}"}
    try:
        result = _OPS[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}


def _validate_positive(**kwargs):
    for name, val in kwargs.items():
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
