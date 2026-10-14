import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.benchmark.utils import SubsetSampler
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

print("Training segment indices:", len(benchmark["train_sampler"]))
print("Query indices:", len(benchmark["test_sampler"]))
print("LDS target values:", lds_targets.shape)
print("Subset indices:", subset_indices.shape)

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

print("One-segment, one-query TRAK score shape:", probe_scores.shape)