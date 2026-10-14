import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
print("Expected full TRAK score matrix (not computed):", expected_score_shape)
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


training_probe = DataLoader(
    benchmark["train_dataset"],
    batch_size=1,
    sampler=SubsetSampler([0]),
)
query_probe = DataLoader(
    benchmark["test_dataset"],
    batch_size=1,
    sampler=SubsetSampler([0]),
)

task = AttributionTask(
    loss_func=target_log_odds,
    model=benchmark["model"],
    checkpoints=[str(benchmark["models_half"][0])],
)
attributor = TRAKAttributor(
    task=task,
    correct_probability_func=target_probability,
    proj_params=TRAKProjectionParams(proj_dim=32),
    device=DEVICE,
    regularization=1e-3,
)

with torch.no_grad():
    attributor.cache(training_probe)
    probe_scores = attributor.attribute(query_probe)

assert probe_scores.shape == (1, 1)
print("One-segment, one-query TRAK score shape:", probe_scores.shape)

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

