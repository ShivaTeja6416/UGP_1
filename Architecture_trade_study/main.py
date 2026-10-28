
from __future__ import annotations
import itertools
from typing import Dict, List, Callable, Any, Optional


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------

def enumerate_combinations(decision_variables: Dict[str, List[Dict[str, Any]]]
                            ) -> List[Dict[str, Dict[str, Any]]]:
    """
    decision_variables = {
        "propulsion": [{"name": "mono", "cost": 2, "mass": 5, ...}, {...}],
        "power":      [{"name": "solar", "cost": 3, "mass": 6, ...}, {...}],
    }
    Returns every combination as {"propulsion": <option dict>, "power": <option dict>}
    -- the full cartesian product, generic over however many variables
    and however many options each has.
    """
    if not decision_variables:
        raise ValueError("decision_variables cannot be empty.")
    var_names = list(decision_variables.keys())
    option_lists = [decision_variables[v] for v in var_names]
    for v, opts in zip(var_names, option_lists):
        if not opts:
            raise ValueError(f"Decision variable '{v}' has no options.")

    combos = []
    for combo_tuple in itertools.product(*option_lists):
        combos.append(dict(zip(var_names, combo_tuple)))
    return combos


# --------------------------------------------------------------------------
# Compatibility filtering
# --------------------------------------------------------------------------

def apply_compatibility_rules(combinations: List[Dict[str, Dict[str, Any]]],
                               rules: List[Callable[[Dict[str, Dict[str, Any]]], bool]]
                               ) -> List[Dict[str, Dict[str, Any]]]:
    """
    Keeps only combinations for which every rule returns True. Each
    rule is a function(combo) -> bool. Rules are arbitrary Python code
    you write per trade study -- this function just applies whatever
    list you give it, generically.
    """
    valid = []
    for combo in combinations:
        if all(rule(combo) for rule in rules):
            valid.append(combo)
    return valid


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def aggregate_criteria(combo: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Sums every numeric field (excluding 'name') across all selected
    options in a combination. E.g. if propulsion contributes cost=2 and
    power contributes cost=3, aggregated cost = 5.
    """
    totals: Dict[str, float] = {}
    for var_name, option in combo.items():
        for key, val in option.items():
            if key == "name":
                continue
            if isinstance(val, (int, float)):
                totals[key] = totals.get(key, 0.0) + val
    return totals


def weighted_score(aggregated_criteria: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Weighted sum: score = sum(weight[c] * aggregated_criteria[c]) over
    every criterion you supply a weight for. Sign convention is yours:
    use a NEGATIVE weight for a criterion where lower is better (e.g.
    cost, mass), positive where higher is better (e.g. reliability).
    """
    return sum(weights.get(c, 0.0) * aggregated_criteria.get(c, 0.0) for c in weights)


def score_combination(combo: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> float:
    """Convenience: aggregate_criteria + weighted_score in one call."""
    return weighted_score(aggregate_criteria(combo), weights)


# --------------------------------------------------------------------------
# Full trade study
# --------------------------------------------------------------------------

def run_trade_study(decision_variables: Dict[str, List[Dict[str, Any]]],
                     weights: Dict[str, float],
                     compatibility_rules: Optional[List[Callable]] = None,
                     top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    One-call convenience: enumerate every combination, filter by
    compatibility rules (if any), score every survivor, and return
    them ranked highest score first.

    Returns a list of dicts: {"combo": {...}, "aggregated_criteria": {...}, "score": float}
    """
    combos = enumerate_combinations(decision_variables)
    if compatibility_rules:
        combos = apply_compatibility_rules(combos, compatibility_rules)
    if not combos:
        raise ValueError("No combinations survived the compatibility rules.")

    scored = []
    for combo in combos:
        crit = aggregate_criteria(combo)
        score = weighted_score(crit, weights)
        scored.append({"combo": combo, "aggregated_criteria": crit, "score": score})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_n] if top_n else scored


# --------------------------------------------------------------------------
# Generic query dispatcher -- ONLY for the plain-data operations.
# run_trade_study / apply_compatibility_rules need real callables and
# are called directly in Python, not through this dispatcher.
# --------------------------------------------------------------------------

def solve(query: dict) -> dict:
    op = query.get("operation")
    inputs = query.get("inputs", {})

    handlers = {
        "aggregate_criteria": aggregate_criteria,
        "weighted_score": weighted_score,
        "score_combination": score_combination,
        "enumerate_combinations": enumerate_combinations,
    }

    if op not in handlers:
        return {"operation": op,
                "error": f"Unknown operation, or this operation requires Python "
                         f"callables (compatibility rules) and must be called "
                         f"directly rather than through solve(). Valid solve() "
                         f"ops: {list(handlers)}. For a full trade study, call "
                         f"run_trade_study() directly."}
    try:
        result = handlers[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}
