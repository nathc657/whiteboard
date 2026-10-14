"""Reproduce the ARIA paper's symbolic Grad-Dot benchmark on MAESTRO."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aria_diagnostics import aria_matrix_diagnostics
from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.load import load_benchmark
from dattri.benchmark.utils import SubsetSampler
from dattri.metric import lds
from dattri.params.projection import TracInProjectionParams
from dattri.task import AttributionTask
from dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNLOAD_PATH = r"C:\Users\nated\.dattri"
GRADDOT_WEIGHT = 1e-3


parser = argparse.ArgumentParser()
parser.add_argument(
    "--full",
    action="store_true",
    help="Compute all 5,000 x 178 Grad-Dot scores and benchmark LDS.",
)
parser.add_argument("--smoke-size", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

print("Using device:", DEVICE)
print(
    "Mode:",
    "full reproduction" if args.full else f"{args.smoke_size}-by-{args.smoke_size} smoke test",
)

benchmark, ground_truth = load_benchmark(
    model="musictransformer",
    dataset="maestro",
    metric="lds",
    download_path=DOWNLOAD_PATH,
)
model = benchmark["model"]
model.device = DEVICE
model = model.to(DEVICE).eval()

train_dataset = benchmark["train_dataset"]
test_dataset = benchmark["test_dataset"]

# Prevent offset-zero segments from being replaced with random crops.
train_dataset.random_seq = False
test_dataset.random_seq = False

train_count = len(benchmark["train_sampler"])
test_count = len(benchmark["test_sampler"])
if args.full:
    train_sampler = benchmark["train_sampler"]
    test_sampler = benchmark["test_sampler"]
    batch_size = args.batch_size
else:
    train_sampler = SubsetSampler(range(min(args.smoke_size, train_count)))
    test_sampler = SubsetSampler(range(min(args.smoke_size, test_count)))
    batch_size = args.smoke_size

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=train_sampler,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=test_sampler,
)


def final_token_loss(params, batch):
    input_sequence, target_sequence = batch
    logits = torch.func.functional_call(
        model,
        params,
        (input_sequence.unsqueeze(0),),
    )
    return F.cross_entropy(
        logits[:, -1, :],
        target_sequence[-1].reshape(1),
        ignore_index=TOKEN_PAD,
        reduction="none",
    ).squeeze(0)


checkpoint_path = benchmark["models_full"][0]
task = AttributionTask(
    model=model,
    loss_func=final_token_loss,
    checkpoints=[str(checkpoint_path)],
)
attributor = TracInAttributor(
    task=task,
    weight_list=torch.tensor([GRADDOT_WEIGHT]),
    normalized_grad=False,
    proj_params=TracInProjectionParams(proj_dim=512),
    device=DEVICE,
)

with torch.no_grad():
    scores = attributor.attribute(train_loader, test_loader)

expected_shape = (
    (train_count, test_count)
    if args.full
    else (len(train_sampler), len(test_sampler))
)
assert scores.shape == expected_shape
print("Grad-Dot score matrix shape:", tuple(scores.shape))

if args.full:
    targets, subset_indices = ground_truth
    lds_values, lds_p_values = lds(scores, ground_truth)
    valid_lds = lds_values[~torch.isnan(lds_values)]
    mean_lds = valid_lds.mean().item() if valid_lds.numel() else float("nan")
    diagnostics = aria_matrix_diagnostics(scores)

    print("Valid LDS queries:", valid_lds.numel(), "of", test_count)
    print("Mean LDS:", mean_lds)
    print("ARIA diagnostics:", diagnostics)

    results_path = Path("output") / "aria_graddot_results.pt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": scores,
            "lds": lds_values,
            "lds_p_values": lds_p_values,
            "targets": targets,
            "subset_indices": subset_indices,
            "aria_diagnostics": diagnostics,
            "checkpoint": str(checkpoint_path),
            "projection_dim": attributor.proj_params.proj_dim,
            "weight": GRADDOT_WEIGHT,
            "random_seq": False,
            "device": DEVICE,
        },
        results_path,
    )
    print("Saved results to:", results_path)
