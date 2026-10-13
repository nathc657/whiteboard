# dattri CPU setup

Working repository: `C:\School\Capstone Folder\whiteboard`.
Upstream dattri is a Git submodule at `vendor/dattri`; keep experiments outside that directory.

## Open and activate

```powershell
cd 'C:\School\Capstone Folder\whiteboard'
.\.venv\Scripts\Activate.ps1
python --version
python -m pip check
python scripts/smoke_music.py
```

In VS Code select **Python: Select Interpreter**, then `.venv\Scripts\python.exe`. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly.

## Recreate on Windows

Install Python 3.12 first. The current environment uses Python 3.12.14.

```powershell
git submodule update --init --recursive
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ./vendor/dattri --no-deps
```

The lock is Windows/Python 3.12 specific; CPU wheels require the PyTorch CPU index above. Build a separate environment and lock for future CUDA work.

## MusicTransformer / MAESTRO source map

- `vendor/dattri/dattri/benchmark/datasets/maestro/__init__.py`: dataset and model factories.
- `vendor/dattri/dattri/benchmark/datasets/maestro/maestro_musictransformer.py`: training and loss evaluation. Inspected defaults: six layers, eight heads, model width 512, feedforward width 1024, sequence length 256, relative positional attention.
- `vendor/dattri/dattri/benchmark/models/MusicTransformer/music_transformer.py`: model implementation.
- `vendor/dattri/dattri/benchmark/models/MusicTransformer/dataset/preprocess_midi.py`: MIDI preprocessing.
- `vendor/dattri/dattri/benchmark/models/MusicTransformer/dataset/e_piano.py`: token sequences and segment sampling.
- `vendor/dattri/dattri/benchmark/load.py`: benchmark loader and remote model/ground-truth artifact URLs. Its imports include unrelated benchmark dependencies; focused MusicTransformer imports are used for this setup.
- `vendor/dattri/test/dattri/benchmark/test_maestro.py`: upstream model and training tests. Some train a full-sized model; use our reduced smoke check first.

## Verification scope

`scripts/smoke_music.py` runs a reduced CPU model forward pass, loss, backward pass, optimizer update, and synthetic segmentation checks. It writes `output/music_smoke.json`. This establishes local code-path compatibility; it does not reproduce pretrained attribution results or establish full benchmark runtime.

Datasets and pretrained checkpoints are not downloaded during setup. Before GPU experiments, obtain the MAESTRO MIDI data and matching full/half checkpoint artifacts, preserve their versions and sample order, and validate the full scoring protocol.

The synthetic segmentation check documents the inspected upstream behavior: offset-zero rows can use random crops. Preserve or deliberately control this behavior when later matching historical ground truth.

## Recorded setup result

- dattri revision: `ef50a35e6cdfa358ca104b7655ef15dcb9650afe`.
- Python: 3.12.14; PyTorch: 2.5.1+cpu; CUDA unavailable (expected).
- Dependency consistency: `pip check` passed.
- Reduced MusicTransformer: output shape `[2, 16, 390]`; loss 6.145303; finite gradients and optimizer update passed.
- Synthetic dataset segmentation checks passed.
- Upstream submodule has no tracked modifications after setup.
- No full benchmark execution or full upstream test suite was run.
