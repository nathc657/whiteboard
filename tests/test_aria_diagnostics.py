import math
from pathlib import Path

import pytest
import torch

from aria_diagnostics import aria_matrix_diagnostics


def _orthogonal_zero_mean_patterns(pattern_count: int) -> torch.Tensor:
    hadamard_two = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    hadamard = torch.ones((1, 1))
    while hadamard.shape[0] < 8:
        hadamard = torch.kron(hadamard, hadamard_two)
    return hadamard[:, 1 : pattern_count + 1] / (8**0.5)


def test_query_specific_matrix_has_distinct_query_profiles():
    scores = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ],
    )

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(0.0)
    assert result["r1"] == pytest.approx(0.5)
    assert result["r2_5"] == pytest.approx(0.5)
    assert result["p"] == pytest.approx(0.0)
    assert result["valid_correlation_pairs"] == 6


def test_query_collapsed_matrix_is_rank_one():
    shared_profile = torch.tensor([0.0, 1.0, 3.0, 6.0])
    scores = torch.stack((shared_profile, 2 * shared_profile, -shared_profile), dim=1)

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(1.0)
    assert result["r1"] == pytest.approx(1.0)
    assert result["r2_5"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] == pytest.approx(0.5434782608695652)


def test_constant_columns_report_undefined_correlation():
    result = aria_matrix_diagnostics(torch.full((4, 3), 2.0))

    assert math.isnan(result["kappa"])
    assert result["r1"] == pytest.approx(1.0)
    assert result["r2_5"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] == pytest.approx(1.0)
    assert result["valid_correlation_pairs"] == 0


def test_zero_matrix_reports_undefined_energy_metrics():
    result = aria_matrix_diagnostics(torch.zeros((4, 3)))

    assert math.isnan(result["kappa"])
    assert math.isnan(result["r1"])
    assert math.isnan(result["r2_5"])
    assert math.isnan(result["p"])
    assert result["valid_correlation_pairs"] == 0
    assert result["nonzero_energy_queries"] == 0


@pytest.mark.parametrize(
    "scores",
    [torch.tensor([1.0, 2.0]), torch.empty((0, 2)), torch.empty((2, 0))],
)
def test_requires_nonempty_matrix(scores):
    with pytest.raises(ValueError, match="non-empty 2D tensor"):
        aria_matrix_diagnostics(scores)


def test_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite values"):
        aria_matrix_diagnostics(torch.tensor([[1.0, float("nan")], [2.0, 3.0]]))


def test_kappa_tracks_shared_energy_from_zero_to_full():
    patterns = _orthogonal_zero_mean_patterns(4)
    common = patterns[:, 0]
    query_specific = patterns[:, 1:4]

    for shared_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = (
            shared_fraction**0.5 * common[:, None]
            + (1.0 - shared_fraction) ** 0.5 * query_specific
        )
        result = aria_matrix_diagnostics(scores)
        assert result["kappa"] == pytest.approx(shared_fraction, abs=1e-6)


def test_random_matrix_matches_theoretical_baseline():
    generator = torch.Generator().manual_seed(20261014)
    scores = torch.randn((5000, 178), generator=generator)

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(0.011, abs=0.001)
    assert result["r1"] == pytest.approx(0.008, abs=0.002)
    assert result["r2_5"] == pytest.approx(0.031, abs=0.008)
    assert result["p"] < 0.002


def test_mean_offset_raises_p_without_query_collapse():
    patterns = _orthogonal_zero_mean_patterns(4)
    offsets = torch.tensor([2.0, -2.0, 2.0, -2.0])
    scores = patterns[:, :4] + torch.ones((8, 1)) * offsets

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] > 0.95


def test_five_shared_patterns_raise_r2_to_5():
    patterns = _orthogonal_zero_mean_patterns(5)
    scores = torch.cat([patterns[:, index : index + 1].repeat(1, 2) for index in range(5)], dim=1)

    result = aria_matrix_diagnostics(scores)

    assert result["r1"] == pytest.approx(0.2, abs=1e-12)
    assert result["r2_5"] == pytest.approx(0.8, abs=1e-12)


def test_kappa_is_invariant_to_per_query_scale():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    factors = torch.tensor([2.0, 3.0, 0.5])

    assert aria_matrix_diagnostics(scores * factors)["kappa"] == pytest.approx(
        aria_matrix_diagnostics(scores)["kappa"],
    )


def test_kappa_is_invariant_to_per_query_offsets():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    offsets = torch.tensor([10.0, -20.0, 7.0])

    assert aria_matrix_diagnostics(scores + offsets)["kappa"] == pytest.approx(
        aria_matrix_diagnostics(scores)["kappa"],
    )


def test_kappa_is_invariant_to_per_query_sign_flips():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    signs = torch.tensor([1.0, -1.0, -1.0])

    assert aria_matrix_diagnostics(scores * signs)["kappa"] == pytest.approx(
        aria_matrix_diagnostics(scores)["kappa"],
    )


def test_all_diagnostics_are_invariant_to_global_scale():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    baseline = aria_matrix_diagnostics(scores)
    scaled = aria_matrix_diagnostics(13.0 * scores)

    for metric in ("kappa", "r1", "r2_5", "p"):
        assert scaled[metric] == pytest.approx(baseline[metric], abs=1e-12)


def test_diagnostics_are_invariant_to_row_and_column_order():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    baseline = aria_matrix_diagnostics(scores)
    reordered = aria_matrix_diagnostics(scores[[2, 0, 3, 1]][:, [2, 0, 1]])

    for metric in ("kappa", "r1", "r2_5", "p"):
        assert reordered[metric] == pytest.approx(baseline[metric], abs=1e-12)


def test_float32_and_float64_inputs_agree():
    scores = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
    )
    single_precision = aria_matrix_diagnostics(scores.float())
    double_precision = aria_matrix_diagnostics(scores.double())

    for metric in ("kappa", "r1", "r2_5", "p"):
        assert single_precision[metric] == pytest.approx(double_precision[metric], abs=1e-12)


def test_mixed_constant_column_is_excluded_from_kappa():
    patterns = _orthogonal_zero_mean_patterns(2)
    scores = torch.stack((patterns[:, 0], torch.full((8,), 3.0), patterns[:, 1]), dim=1)

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] == pytest.approx(1 / 3)
    assert result["valid_correlation_pairs"] == 2
    assert result["nonzero_energy_queries"] == 3


def test_single_query_has_undefined_kappa():
    result = aria_matrix_diagnostics(torch.tensor([[1.0], [-1.0], [0.0], [2.0]]))

    assert math.isnan(result["kappa"])
    assert result["valid_correlation_pairs"] == 0


def test_zero_energy_query_is_excluded_from_p():
    patterns = _orthogonal_zero_mean_patterns(2)
    scores = torch.stack((patterns[:, 0], torch.zeros(8), patterns[:, 1]), dim=1)

    result = aria_matrix_diagnostics(scores)

    assert result["kappa"] == pytest.approx(0.0, abs=1e-12)
    assert result["p"] == pytest.approx(0.0, abs=1e-12)
    assert result["valid_correlation_pairs"] == 2
    assert result["nonzero_energy_queries"] == 2


REAL_RESULTS_PATH = Path("output/aria_graddot_results.pt")


@pytest.mark.skipif(
    not REAL_RESULTS_PATH.exists(),
    reason="full Grad-Dot reproduction results are not present",
)
def test_real_grad_dot_matrix_matches_report():
    results = torch.load(REAL_RESULTS_PATH, map_location="cpu", weights_only=True)
    scores = results["scores"]
    diagnostics = aria_matrix_diagnostics(scores)
    valid_lds = results["lds"][~torch.isnan(results["lds"])]

    assert scores.shape == (5000, 178)
    assert valid_lds.mean().item() == pytest.approx(0.088, abs=0.002)
    assert diagnostics["kappa"] == pytest.approx(0.129, abs=0.002)
    assert diagnostics["r1"] == pytest.approx(0.147, abs=0.002)
    assert diagnostics["r2_5"] == pytest.approx(0.239, abs=0.002)
    assert diagnostics["p"] == pytest.approx(0.035, abs=0.002)
