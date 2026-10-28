
import statistics
from main import (
    filter_by_class, z_score, percentile_rank, is_outlier_zscore,
    is_outlier_iqr, nearest_neighbors, benchmark_design, solve,
)


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.4f} expected={expected:.4f} (tol={tol})")
    assert ok, f"{name} FAILED"


# --- Test 1: z-score, hand calc
# population=[10,12,14,16,18] -> mean=14, sample stdev=sqrt(10)=3.16228
# value=20 -> z = (20-14)/3.16228 = 1.8974
def test_z_score():
    pop = [10, 12, 14, 16, 18]
    z = z_score(20, pop)
    mean = 14
    stdev = statistics.stdev(pop)  # sample stdev, matches module's method choice
    expected = (20 - mean) / stdev
    check("z_score", z, expected, tol=1e-6)


# --- Test 2: percentile rank, hand calc
# population=[10,20,30,40,50], value=30 -> 3 of 5 values <= 30 -> 60%
def test_percentile_rank():
    pop = [10, 20, 30, 40, 50]
    p = percentile_rank(30, pop)
    check("percentile_rank_30", p, 60.0, tol=1e-6)
    p2 = percentile_rank(25, pop)
    check("percentile_rank_25", p2, 40.0, tol=1e-6)


# --- Test 3: z-score outlier flag, clear-cut case
def test_is_outlier_zscore():
    pop = [10, 12, 14, 16, 18]
    result = is_outlier_zscore(30, pop, threshold=2.0)
    print(f"[{'PASS' if result['is_outlier'] else 'FAIL'}] "
          f"clear outlier (30 vs tight cluster around 14) flagged: {result['is_outlier']}")
    assert result["is_outlier"] is True

    result2 = is_outlier_zscore(15, pop, threshold=2.0)
    print(f"[{'PASS' if not result2['is_outlier'] else 'FAIL'}] "
          f"non-outlier (15, inside cluster) flagged: {result2['is_outlier']}")
    assert result2["is_outlier"] is False


# --- Test 4: IQR outlier flag, sanity check (exact fence values depend
# on the quantile method, so we check the qualitative outcome, not an
# exact number)
def test_is_outlier_iqr():
    pop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    result_extreme = is_outlier_iqr(100, pop)
    result_normal = is_outlier_iqr(5, pop)
    print(f"[{'PASS' if result_extreme['is_outlier'] else 'FAIL'}] "
          f"extreme value (100) flagged: {result_extreme['is_outlier']}")
    print(f"[{'PASS' if not result_normal['is_outlier'] else 'FAIL'}] "
          f"normal value (5) flagged: {result_normal['is_outlier']}")
    assert result_extreme["is_outlier"] is True
    assert result_normal["is_outlier"] is False


# --- Test 5: nearest neighbors, single feature so raw distance ordering
# is predictable by hand (z-score normalization is monotonic in 1D, so
# it can't reorder points -- only matters when combining multiple
# features on different scales).
# Population mass_kg: A=500, B=600, C=550, D=700, E=450
# Query mass_kg=560 -> |diff|: A=60, B=40, C=10, D=140, E=110
# Expected nearest 3, in order: C, B, A
def test_nearest_neighbors_single_feature():
    population = [
        {"name": "A", "mass_kg": 500},
        {"name": "B", "mass_kg": 600},
        {"name": "C", "mass_kg": 550},
        {"name": "D", "mass_kg": 700},
        {"name": "E", "mass_kg": 450},
    ]
    results = nearest_neighbors({"mass_kg": 560}, population, k=3)
    names = [r.entry["name"] for r in results]
    print(f"[{'PASS' if names == ['C', 'B', 'A'] else 'FAIL'}] "
          f"nearest-3 order: {names} (expected ['C', 'B', 'A'])")
    assert names == ["C", "B", "A"]


# --- Test 6: mission-class filtering
def test_filter_by_class():
    population = [
        {"name": "A", "mass_kg": 500, "mission_class": "lander"},
        {"name": "B", "mass_kg": 600, "mission_class": "lander"},
        {"name": "F", "mass_kg": 5000, "mission_class": "orbiter"},
    ]
    filtered = filter_by_class(population, "mission_class", "lander")
    check("filter_by_class_count", len(filtered), 2, tol=0)


# --- Test 7: combined benchmark_design call
def test_benchmark_design_combined():
    population = [
        {"name": "A", "mass_kg": 500, "delta_v_mps": 1000, "mission_class": "lander"},
        {"name": "B", "mass_kg": 600, "delta_v_mps": 1100, "mission_class": "lander"},
        {"name": "C", "mass_kg": 550, "delta_v_mps": 1050, "mission_class": "lander"},
        {"name": "D", "mass_kg": 700, "delta_v_mps": 1200, "mission_class": "lander"},
        {"name": "E", "mass_kg": 450, "delta_v_mps": 950, "mission_class": "lander"},
        {"name": "F", "mass_kg": 5000, "delta_v_mps": 3000, "mission_class": "orbiter"},
    ]
    result = benchmark_design(
        design_vector={"mass_kg": 560, "delta_v_mps": 1080},
        population=population,
        class_field="mission_class", class_value="lander",
        k=2,
    )
    check("benchmark_population_size_used", result["population_size_used"], 5, tol=0)
    assert "mass_kg" in result["per_feature_outlier_check"]
    assert "delta_v_mps" in result["per_feature_outlier_check"]
    print(f"[PASS] benchmark_design ran end-to-end, orbiter F correctly excluded "
          f"(used {result['population_size_used']} of 6 total entries)")


# --- Test 8: dispatcher
def test_solve_dispatcher():
    pop = [10, 12, 14, 16, 18]
    r = solve({"operation": "z_score", "inputs": {"value": 20, "population_values": pop}})
    assert r["status"] == "validated"
    expected = (20 - 14) / statistics.stdev(pop)
    check("dispatcher_z_score", r["result"], expected, tol=1e-6)


# --- Test 9: bad input handling
def test_invalid_inputs():
    try:
        z_score(10, [5])  # only 1 population value
        assert False, "expected ValueError for population too small"
    except ValueError:
        print("[PASS] too-small population correctly raised ValueError")


if __name__ == "__main__":
    test_z_score()
    test_percentile_rank()
    test_is_outlier_zscore()
    test_is_outlier_iqr()
    test_nearest_neighbors_single_feature()
    test_filter_by_class()
    test_benchmark_design_combined()
    test_solve_dispatcher()
    test_invalid_inputs()
    print("\nAll tests passed.")
