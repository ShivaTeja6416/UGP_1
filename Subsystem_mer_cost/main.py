
from __future__ import annotations
import math
from typing import Dict, Optional


def _validate_positive(**kwargs):
    for name, val in kwargs.items():
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")


# --------------------------------------------------------------------------
# Illustrative-only fractions. DO NOT treat as authoritative.
# Rough midpoints of commonly-cited small-spacecraft ranges (the kind of
# number you'd find in an intro SMAD-style table), included purely so
# this module has something to run against before your corpus-sourced
# numbers are ready. Replace before trusting any real output.
# --------------------------------------------------------------------------
ILLUSTRATIVE_TYPICAL_FRACTIONS = {
    "structure": 0.20,
    "power": 0.20,
    "adcs": 0.10,
    "thermal": 0.04,
    "comm": 0.05,
    "cdh": 0.04,
    "margin": 0.15,
}


# --------------------------------------------------------------------------
# Percentage-of-total-dry-mass closure
# --------------------------------------------------------------------------

def mass_closure_percentage_method(payload_mass_kg: float,
                                    subsystem_fractions: Dict[str, float]) -> dict:
    """
    Classic early-design mass closure: if every subsystem's mass is a
    fixed fraction of the TOTAL dry mass, and payload_mass is the one
    fixed (non-fractional) input, then:

        total_dry_mass = payload_mass + sum(fraction_i * total_dry_mass)
        total_dry_mass * (1 - sum(fractions)) = payload_mass
        total_dry_mass = payload_mass / (1 - sum(fractions))

    This is solved algebraically (no iteration needed) because the
    percentage-method is linear in total_dry_mass.
    """
    _validate_positive(payload_mass_kg=payload_mass_kg)
    total_fraction = sum(subsystem_fractions.values())
    if total_fraction >= 1:
        raise ValueError(f"Sum of subsystem fractions ({total_fraction}) must be < 1 "
                          f"(payload has to be a positive share of the total).")
    if total_fraction < 0:
        raise ValueError("Subsystem fractions cannot be negative.")

    total_dry_mass = payload_mass_kg / (1 - total_fraction)
    subsystem_masses = {k: v * total_dry_mass for k, v in subsystem_fractions.items()}

    return {
        "payload_mass_kg": payload_mass_kg,
        "total_dry_mass_kg": total_dry_mass,
        "subsystem_masses_kg": subsystem_masses,
    }


# --------------------------------------------------------------------------
# Generic parametric MERs (coefficients supplied, not fitted or assumed)
# --------------------------------------------------------------------------

def mer_power_law(x: float, a: float, b: float) -> float:
    """mass = a * x^b -- generic power-law MER. a and b are YOUR inputs,
    sourced from a real published relationship, e.g. solar array mass
    as a function of required power."""
    _validate_positive(x=x, a=a)
    return a * x ** b


def mer_linear(x: float, a: float, b: float = 0.0) -> float:
    """mass = a*x + b -- generic linear MER."""
    if x < 0:
        raise ValueError("x must be >= 0.")
    return a * x + b


# --------------------------------------------------------------------------
# Cost estimating relationships (CER)
# --------------------------------------------------------------------------

def cost_estimate(mass_kg: float, cost_per_kg: float, fixed_cost: float = 0.0) -> float:
    """Simple linear CER: cost = cost_per_kg * mass + fixed_cost."""
    _validate_positive(mass_kg=mass_kg, cost_per_kg=cost_per_kg)
    if fixed_cost < 0:
        raise ValueError("fixed_cost cannot be negative.")
    return cost_per_kg * mass_kg + fixed_cost


def subsystem_cost_breakdown(subsystem_masses_kg: Dict[str, float],
                              cost_per_kg: Dict[str, float],
                              fixed_costs: Optional[Dict[str, float]] = None) -> dict:
    """
    Per-subsystem cost, given a mass breakdown (e.g. straight from
    mass_closure_percentage_method) and a cost-per-kg for each
    subsystem. Any subsystem missing from cost_per_kg is skipped (with
    a note), rather than silently assumed to cost 0.
    """
    fixed_costs = fixed_costs or {}
    breakdown = {}
    skipped = []
    for subsystem, mass in subsystem_masses_kg.items():
        if subsystem not in cost_per_kg:
            skipped.append(subsystem)
            continue
        breakdown[subsystem] = cost_estimate(
            mass, cost_per_kg[subsystem], fixed_costs.get(subsystem, 0.0)
        )
    return {
        "per_subsystem_cost": breakdown,
        "total_cost": sum(breakdown.values()),
        "skipped_subsystems_no_rate_given": skipped,
    }


# --------------------------------------------------------------------------
# Combined point-design helper
# --------------------------------------------------------------------------

def point_design_estimate(payload_mass_kg: float,
                           subsystem_fractions: Dict[str, float],
                           cost_per_kg: Dict[str, float]) -> dict:
    """
    One-call convenience: mass closure, then cost breakdown on top of
    the resulting subsystem masses, for a full first-pass point design.
    """
    mass_result = mass_closure_percentage_method(payload_mass_kg, subsystem_fractions)
    cost_result = subsystem_cost_breakdown(mass_result["subsystem_masses_kg"], cost_per_kg)
    return {**mass_result, **cost_result}


# --------------------------------------------------------------------------
# Generic query dispatcher
# --------------------------------------------------------------------------

def solve(query: dict) -> dict:
    op = query.get("operation")
    inputs = query.get("inputs", {})

    handlers = {
        "mass_closure_percentage_method": mass_closure_percentage_method,
        "mer_power_law": mer_power_law,
        "mer_linear": mer_linear,
        "cost_estimate": cost_estimate,
        "subsystem_cost_breakdown": subsystem_cost_breakdown,
        "point_design_estimate": point_design_estimate,
    }

    if op not in handlers:
        return {"operation": op, "error": f"Unknown operation. Valid: {list(handlers)}"}
    try:
        result = handlers[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}
