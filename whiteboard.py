import torch
from torch.utils.data import DataLoader, Subset

from dattri.benchmark.datasets.maestro import (
    create_maestro_datasets,
    create_musictransformer_model,
    loss_maestro_musictransformer,
    train_maestro_musictransformer,
)
from dattri.metric import lds


