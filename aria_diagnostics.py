"""Reliability diagnostics for segment-level attribution score matrices."""

import torch


def aria_matrix_diagnostics(score_matrix: torch.Tensor) -> dict[str, float | int]:
    """Compute ARIA's kappa, r1, r2:5, and p diagnostics.

    Constant columns have undefined Pearson correlations and are excluded from
    kappa. Zero-energy columns are excluded from p. The result reports the
    number of valid pairs/queries so undefined cases remain visible.
    """
    if score_matrix.ndim != 2 or min(score_matrix.shape) == 0:
        raise ValueError("score_matrix must be a non-empty 2D tensor")

    matrix = score_matrix.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("score_matrix must contain only finite values")

    segment_count, _ = matrix.shape
    column_means = matrix.mean(dim=0)
    column_energies = matrix.square().sum(dim=0)
    total_energy = column_energies.sum()

    if total_energy > 0:
        singular_values = torch.linalg.svdvals(matrix)
        energy_ratios = singular_values.square() / total_energy
        r1 = energy_ratios[0].item()
        r2_5 = energy_ratios[1:5].sum().item()
    else:
        r1 = float("nan")
        r2_5 = float("nan")

    nonzero_columns = column_energies > 0
    p = (
        (
            segment_count
            * column_means[nonzero_columns].square()
            / column_energies[nonzero_columns]
        )
        .mean()
        .item()
        if nonzero_columns.any()
        else float("nan")
    )

    centered = matrix - column_means
    centered_norms = torch.linalg.vector_norm(centered, dim=0)
    nonconstant_columns = centered_norms > 0
    valid_column_count = int(nonconstant_columns.sum().item())
    valid_correlation_pairs = valid_column_count * (valid_column_count - 1)

    if valid_column_count >= 2:
        normalized = centered[:, nonconstant_columns] / centered_norms[nonconstant_columns]
        correlations = normalized.T @ normalized
        off_diagonal = ~torch.eye(valid_column_count, dtype=torch.bool)
        kappa = correlations[off_diagonal].abs().mean().item()
    else:
        kappa = float("nan")

    return {
        "kappa": kappa,
        "r1": r1,
        "r2_5": r2_5,
        "p": p,
        "valid_correlation_pairs": valid_correlation_pairs,
        "nonzero_energy_queries": int(nonzero_columns.sum().item()),
    }
