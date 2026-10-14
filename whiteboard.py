from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aria_diagnostics import aria_matrix_diagnostics
from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.benchmark.utils import SubsetSampler
from dattri.metric import lds
from dattri.params.projection import TRAKProjectionParams
from dattri.task import AttributionTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

benchmark, (lds_targets, subset_indices) = load_benchmark(
    model="musictransformer",
    dataset="maestro",
    metric="lds",
    download_path=r"C:\Users\nated\.dattri",
)
benchmark["model"].device = DEVICE
benchmark["model"] = benchmark["model"].to(DEVICE)

training_count = len(benchmark["train_sampler"])
query_count = len(benchmark["test_sampler"])
subset_count, subset_size = subset_indices.shape
expected_score_shape = (training_count, query_count)

assert lds_targets.shape == (subset_count, query_count)
assert subset_indices.max().item() < training_count

print("Training segment indices:", training_count)
print("Query indices:", query_count)
print("Expected full TRAK score matrix:", expected_score_shape)
print("LDS target values:", lds_targets.shape)
print("Subset indices:", subset_indices.shape)
print("Each LDS subset uses this many training segments:", subset_size)

train_input, train_target = benchmark["train_dataset"][0]
query_input, query_target = benchmark["test_dataset"][0]

print("Training example tensors:", train_input.shape, train_target.shape)
print("Query example tensors:", query_input.shape, query_target.shape)


def target_log_probability(params, batch):
    input_sequence, target_sequence = batch
    logits = torch.func.functional_call(
        benchmark["model"],
        params,
        (input_sequence.unsqueeze(0),),
    )
    log_probabilities = F.log_softmax(logits[0, -1], dim=-1)
    target_index = target_sequence[-1].reshape(1)
    return log_probabilities.gather(0, target_index).squeeze(0)


def target_log_odds(params, batch):
    log_probability = target_log_probability(params, batch)
    return log_probability - torch.log1p(-log_probability.exp())


def target_probability(params, batch):
    return target_log_probability(params, batch).exp()


RUN_FULL_TRAK = False
BATCH_SIZE = 8
checkpoint_path = benchmark["models_full"][0]

if RUN_FULL_TRAK:
    print(
        "Using full-data checkpoint 0 of",
        len(benchmark["models_full"]),
        "for this single-checkpoint matrix.",
    )
    training_loader = DataLoader(
        benchmark["train_dataset"],
        batch_size=BATCH_SIZE,
        sampler=benchmark["train_sampler"],
    )
    query_loader = DataLoader(
        benchmark["test_dataset"],
        batch_size=BATCH_SIZE,
        sampler=benchmark["test_sampler"],
    )
else:
    print("Full TRAK matrix disabled; running one-pair probe only.")
    training_loader = DataLoader(
        benchmark["train_dataset"],
        batch_size=1,
        sampler=SubsetSampler([0]),
    )
    query_loader = DataLoader(
        benchmark["test_dataset"],
        batch_size=1,
        sampler=SubsetSampler([0]),
    )

task = AttributionTask(
    loss_func=target_log_odds,
    model=benchmark["model"],
    checkpoints=[str(checkpoint_path)],
)
attributor = TRAKAttributor(
    task=task,
    correct_probability_func=target_probability,
    proj_params=TRAKProjectionParams(),
    device=DEVICE,
    regularization=1e-3,
)

with torch.no_grad():
    attributor.cache(training_loader)
    trak_scores = attributor.attribute(query_loader)

expected_run_shape = expected_score_shape if RUN_FULL_TRAK else (1, 1)
assert trak_scores.shape == expected_run_shape
print("TRAK score matrix shape:", trak_scores.shape)

if RUN_FULL_TRAK:
    aria_metrics = aria_matrix_diagnostics(trak_scores)
    lds_correlations, lds_p_values = lds(
        trak_scores,
        (lds_targets, subset_indices),
    )
    valid_lds = lds_correlations[~torch.isnan(lds_correlations)]
    mean_lds = valid_lds.mean() if valid_lds.numel() else torch.tensor(float("nan"))

    print("Valid per-query LDS values:", valid_lds.numel(), "of", query_count)
    print("Mean LDS:", mean_lds.item())
    print("ARIA score-matrix diagnostics:", aria_metrics)

    results_path = Path("output") / "maestro_trak_lds_single_full_checkpoint.pt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": trak_scores.cpu(),
            "lds_correlations": lds_correlations.cpu(),
            "lds_p_values": lds_p_values.cpu(),
            "targets": lds_targets.cpu(),
            "subset_indices": subset_indices.cpu(),
            "aria_diagnostics": aria_metrics,
            "checkpoint": str(checkpoint_path),
            "projection_dim": attributor.proj_params.proj_dim,
            "device": DEVICE,
        },
        results_path,
    )
    print("Saved full matrix and LDS results to:", results_path)

synthetic_scores = torch.tensor(
    [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]],
)
synthetic_targets = torch.tensor(
    [[0.1, 0.3], [0.2, 0.2], [0.3, 0.1]],
)
synthetic_subset_indices = torch.tensor(
    [[0, 1], [0, 2], [1, 2]],
)
subset_score_sums = synthetic_scores[synthetic_subset_indices].sum(dim=1)
print("Synthetic subset score sums:", subset_score_sums)

# LDS compares each query's subset score sums with its target values by rank.
synthetic_lds, _ = lds(
    synthetic_scores,
    (synthetic_targets, synthetic_subset_indices),
)
assert torch.allclose(synthetic_lds, torch.ones(2))
print("Synthetic LDS correlations:", synthetic_lds)

reversed_targets = synthetic_targets.flip(0)
reversed_lds, _ = lds(
    synthetic_scores,
    (reversed_targets, synthetic_subset_indices),
)
assert torch.allclose(reversed_lds, -torch.ones(2))
print("LDS with reversed target rankings:", reversed_lds)

partial_scores = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
partial_targets = torch.tensor([[0.1], [0.3], [0.2], [0.4]])
single_sample_subsets = torch.arange(4).reshape(-1, 1)
partial_lds, _ = lds(
    partial_scores,
    (partial_targets, single_sample_subsets),
)
assert torch.allclose(partial_lds, torch.tensor([0.8]))
print("LDS with partial rank agreement:", partial_lds)

aria_query_specific = torch.tensor(
    [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]],
)
query_specific_diagnostics = aria_matrix_diagnostics(aria_query_specific)
assert abs(query_specific_diagnostics["kappa"]) < 1e-12
assert abs(query_specific_diagnostics["r1"] - 0.5) < 1e-12
assert abs(query_specific_diagnostics["r2_5"] - 0.5) < 1e-12
assert abs(query_specific_diagnostics["p"]) < 1e-12
print("ARIA query-specific diagnostics:", query_specific_diagnostics)

shared_profile = torch.tensor([0.0, 1.0, 3.0, 6.0])
aria_query_collapsed = torch.stack(
    (shared_profile, 2 * shared_profile, -shared_profile),
    dim=1,
)
collapsed_diagnostics = aria_matrix_diagnostics(aria_query_collapsed)
assert abs(collapsed_diagnostics["kappa"] - 1.0) < 1e-12
assert abs(collapsed_diagnostics["r1"] - 1.0) < 1e-12
assert abs(collapsed_diagnostics["r2_5"]) < 1e-12
print("ARIA query-collapsed diagnostics:", collapsed_diagnostics)

aria_constant = aria_matrix_diagnostics(torch.full((4, 3), 2.0))
assert torch.isnan(torch.tensor(aria_constant["kappa"]))
assert abs(aria_constant["r1"] - 1.0) < 1e-12
assert abs(aria_constant["p"] - 1.0) < 1e-12
print("ARIA constant-column diagnostics:", aria_constant)

