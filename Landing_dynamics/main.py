
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

G0 = 9.80665  # m/s^2, standard gravity (used to convert "g's" to m/s^2)


def _validate_positive(**kwargs):
    for name, val in kwargs.items():
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")


# --------------------------------------------------------------------------
# Core energy / kinematics
# --------------------------------------------------------------------------

def impact_energy(mass_kg: float, velocity_mps: float) -> float:
    """Kinetic energy at touchdown: KE = 0.5 * m * v^2  [J]"""
    _validate_positive(mass_kg=mass_kg)
    if velocity_mps < 0:
        raise ValueError("velocity_mps must be >= 0.")
    return 0.5 * mass_kg * velocity_mps ** 2


def required_stroke_length(velocity_mps: float, allowable_decel_g: float) -> float:
    """
    Minimum crush stroke needed to decelerate from velocity_mps to 0
    without exceeding allowable_decel_g, assuming CONSTANT deceleration
    (idealized -- real crushable materials aren't perfectly constant-
    force, but this is the standard first-pass sizing equation):

        v^2 = 2 * a * s   =>   s = v^2 / (2*a)
        a = allowable_decel_g * g0
    """
    _validate_positive(allowable_decel_g=allowable_decel_g)
    if velocity_mps < 0:
        raise ValueError("velocity_mps must be >= 0.")
    a = allowable_decel_g * G0
    return velocity_mps ** 2 / (2 * a)


def average_impact_force(mass_kg: float, allowable_decel_g: float) -> float:
    """Average force experienced during a constant-deceleration crush: F = m*a"""
    _validate_positive(mass_kg=mass_kg, allowable_decel_g=allowable_decel_g)
    return mass_kg * allowable_decel_g * G0


def stroke_from_force_and_energy(force_N: float, energy_J: float) -> float:
    """Inverse of work = force * distance: s = energy / force"""
    _validate_positive(force_N=force_N, energy_J=energy_J)
    return energy_J / force_N


def deceleration_from_stroke(velocity_mps: float, stroke_m: float) -> float:
    """
    Inverse of required_stroke_length: given a fixed available stroke,
    what deceleration (in g's) results?
        a = v^2 / (2*s)
    """
    _validate_positive(stroke_m=stroke_m)
    if velocity_mps < 0:
        raise ValueError("velocity_mps must be >= 0.")
    a_mps2 = velocity_mps ** 2 / (2 * stroke_m)
    return a_mps2 / G0


# --------------------------------------------------------------------------
# Attenuator mass sizing
# --------------------------------------------------------------------------

def attenuator_mass(energy_to_absorb_J: float, sea_j_per_kg: float) -> float:
    """
    Mass of crushable material needed to absorb a given amount of energy,
    given the material's specific energy absorption (SEA), in J/kg.
    SEA is a material property you supply (e.g. from a datasheet or your
    document corpus) -- this module does not assume a value.
        m_attenuator = energy_to_absorb / SEA
    """
    _validate_positive(energy_to_absorb_J=energy_to_absorb_J, sea_j_per_kg=sea_j_per_kg)
    return energy_to_absorb_J / sea_j_per_kg


# --------------------------------------------------------------------------
# Combined design helper
# --------------------------------------------------------------------------

@dataclass
class AttenuatorDesign:
    impact_energy_J: float
    stroke_length_m: float
    avg_force_N: float
    attenuator_mass_kg: float


def design_attenuator(mass_kg: float, velocity_mps: float,
                       allowable_decel_g: float, sea_j_per_kg: float) -> AttenuatorDesign:
    """
    One-call convenience: given a landed mass, touchdown velocity, an
    allowable deceleration limit, and a candidate material's SEA,
    returns the full first-pass sizing: energy to absorb, stroke needed,
    average force, and attenuator mass.
    """
    ke = impact_energy(mass_kg, velocity_mps)
    s = required_stroke_length(velocity_mps, allowable_decel_g)
    f = average_impact_force(mass_kg, allowable_decel_g)
    m_att = attenuator_mass(ke, sea_j_per_kg)
    return AttenuatorDesign(
        impact_energy_J=ke, stroke_length_m=s,
        avg_force_N=f, attenuator_mass_kg=m_att,
    )


# --------------------------------------------------------------------------
# Generic query dispatcher (same convention as the other tools)
# --------------------------------------------------------------------------

_OPS = {
    "impact_energy": impact_energy,
    "required_stroke_length": required_stroke_length,
    "average_impact_force": average_impact_force,
    "stroke_from_force_and_energy": stroke_from_force_and_energy,
    "deceleration_from_stroke": deceleration_from_stroke,
    "attenuator_mass": attenuator_mass,
}


def solve(query: dict) -> dict:
    op = query.get("operation")
    inputs = query.get("inputs", {})

    if op == "design_attenuator":
        try:
            result = design_attenuator(**inputs)
            return {
                "operation": op, "inputs": inputs,
                "result": {
                    "impact_energy_J": result.impact_energy_J,
                    "stroke_length_m": result.stroke_length_m,
                    "avg_force_N": result.avg_force_N,
                    "attenuator_mass_kg": result.attenuator_mass_kg,
                },
                "status": "validated",
            }
        except (TypeError, ValueError) as e:
            return {"operation": op, "inputs": inputs, "error": str(e)}

    if op not in _OPS:
        return {"operation": op, "error": f"Unknown operation. Valid: {list(_OPS) + ['design_attenuator']}"}
    try:
        result = _OPS[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}
