import math

import pytest
import torch

from aria_diagnostics import aria_matrix_diagnostics


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
