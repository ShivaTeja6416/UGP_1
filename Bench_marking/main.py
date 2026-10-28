
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


def _validate_population(population: List[Dict[str, Any]]):
    if not population or len(population) < 2:
        raise ValueError("population must contain at least 2 entries to benchmark against.")


# --------------------------------------------------------------------------
# Mission-class filtering
# --------------------------------------------------------------------------

def filter_by_class(population: List[Dict[str, Any]], class_field: str,
                     class_value: str) -> List[Dict[str, Any]]:
    """
    Keep only population entries where population[i][class_field] ==
    class_value. E.g. filter_by_class(missions, "mission_class", "lander")
    so you never benchmark a lander against an orbiter.
    """
    filtered = [p for p in population if p.get(class_field) == class_value]
    if not filtered:
        raise ValueError(f"No population entries with {class_field}={class_value!r}.")
    return filtered


# --------------------------------------------------------------------------
# Single-feature statistics
# --------------------------------------------------------------------------

def z_score(value: float, population_values: List[float]) -> float:
    """(value - mean) / stdev of the population. stdev computed as
    sample stdev (n-1) since a comparison population is a sample, not
    the full universe of possible designs."""
    if len(population_values) < 2:
        raise ValueError("Need at least 2 population values to compute a z-score.")
    mean = statistics.mean(population_values)
    stdev = statistics.stdev(population_values)
    if stdev == 0:
        raise ValueError("Population has zero variance; z-score is undefined.")
    return (value - mean) / stdev


def percentile_rank(value: float, population_values: List[float]) -> float:
    """
    What percentile does `value` fall at within population_values?
    0 = below everything, 100 = above everything. Uses the standard
    "fraction of population <= value" definition.
    """
    if not population_values:
        raise ValueError("population_values cannot be empty.")
    n = len(population_values)
    n_at_or_below = sum(1 for v in population_values if v <= value)
    return 100.0 * n_at_or_below / n


def is_outlier_zscore(value: float, population_values: List[float],
                       threshold: float = 2.0) -> Dict[str, Any]:
    """Flags value as an outlier if |z-score| > threshold (default 2.0,
    a common convention; caller can tighten/loosen it)."""
    z = z_score(value, population_values)
    return {"z_score": z, "is_outlier": abs(z) > threshold, "threshold": threshold}


def is_outlier_iqr(value: float, population_values: List[float],
                    k: float = 1.5) -> Dict[str, Any]:
    """
    Classic IQR fence method: outlier if value is below Q1 - k*IQR or
    above Q3 + k*IQR. Default k=1.5 is the standard Tukey convention.
    More robust than z-score for small or skewed populations.
    """
    if len(population_values) < 4:
        raise ValueError("Need at least 4 population values for a meaningful IQR.")
    sorted_vals = sorted(population_values)
    q1 = statistics.quantiles(sorted_vals, n=4)[0]
    q3 = statistics.quantiles(sorted_vals, n=4)[2]
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower_fence": lower_fence, "upper_fence": upper_fence,
        "is_outlier": value < lower_fence or value > upper_fence,
    }


# --------------------------------------------------------------------------
# Multi-feature nearest-neighbor ranking
# --------------------------------------------------------------------------

@dataclass
class NeighborResult:
    entry: Dict[str, Any]
    distance: float


def nearest_neighbors(design_vector: Dict[str, float],
                       population: List[Dict[str, Any]],
                       k: int = 3) -> List[NeighborResult]:
    """
    Ranks population entries by Euclidean distance to design_vector,
    across whatever numeric features design_vector specifies. Features
    are z-score normalized using the population's own mean/stdev for
    each feature, so features on very different scales (e.g. mass_kg
    vs delta_v_mps) contribute comparably to the distance -- otherwise
    a feature measured in the thousands would dominate one measured in
    single digits regardless of which actually matters more.

    Returns the k closest entries, nearest first.
    """
    _validate_population(population)
    features = list(design_vector.keys())
    if not features:
        raise ValueError("design_vector must specify at least one feature.")

    # Compute per-feature mean/stdev from the population for normalization
    stats = {}
    for feat in features:
        vals = [p[feat] for p in population if feat in p]
        if len(vals) < 2:
            raise ValueError(f"Feature '{feat}' missing or insufficient in population.")
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 1.0
        stats[feat] = (mean, stdev if stdev != 0 else 1.0)

    def normalized_distance(entry: Dict[str, Any]) -> float:
        sq_sum = 0.0
        for feat in features:
            mean, stdev = stats[feat]
            dv_norm = (design_vector[feat] - mean) / stdev
            entry_norm = (entry[feat] - mean) / stdev
            sq_sum += (dv_norm - entry_norm) ** 2
        return math.sqrt(sq_sum)

    scored = [NeighborResult(entry=p, distance=normalized_distance(p)) for p in population]
    scored.sort(key=lambda r: r.distance)
    return scored[:k]


# --------------------------------------------------------------------------
# Combined benchmark call
# --------------------------------------------------------------------------

def benchmark_design(design_vector: Dict[str, float],
                      population: List[Dict[str, Any]],
                      class_field: Optional[str] = None,
                      class_value: Optional[str] = None,
                      k: int = 3,
                      outlier_threshold: float = 2.0) -> Dict[str, Any]:
    """
    One-call convenience: optionally filter the population by mission
    class, then return nearest neighbors PLUS a per-feature z-score/
    outlier flag for the design vector against the (filtered) population.
    """
    pop = population
    if class_field and class_value:
        pop = filter_by_class(population, class_field, class_value)

    neighbors = nearest_neighbors(design_vector, pop, k=k)

    per_feature = {}
    for feat, val in design_vector.items():
        vals = [p[feat] for p in pop if feat in p]
        per_feature[feat] = is_outlier_zscore(val, vals, threshold=outlier_threshold)

    return {
        "population_size_used": len(pop),
        "nearest_neighbors": [{"entry": n.entry, "distance": n.distance} for n in neighbors],
        "per_feature_outlier_check": per_feature,
    }


# --------------------------------------------------------------------------
# Generic query dispatcher
# --------------------------------------------------------------------------

def solve(query: dict) -> dict:
    op = query.get("operation")
    inputs = query.get("inputs", {})

    handlers = {
        "z_score": z_score,
        "percentile_rank": percentile_rank,
        "is_outlier_zscore": is_outlier_zscore,
        "is_outlier_iqr": is_outlier_iqr,
        "filter_by_class": filter_by_class,
        "benchmark_design": benchmark_design,
    }

    if op == "nearest_neighbors":
        try:
            result = nearest_neighbors(**inputs)
            return {
                "operation": op, "inputs": inputs,
                "result": [{"entry": r.entry, "distance": r.distance} for r in result],
                "status": "validated",
            }
        except (TypeError, ValueError) as e:
            return {"operation": op, "inputs": inputs, "error": str(e)}

    if op not in handlers:
        return {"operation": op, "error": f"Unknown operation. Valid: {list(handlers) + ['nearest_neighbors']}"}
    try:
        result = handlers[op](**inputs)
        return {"operation": op, "inputs": inputs, "result": result, "status": "validated"}
    except (TypeError, ValueError) as e:
        return {"operation": op, "inputs": inputs, "error": str(e)}
