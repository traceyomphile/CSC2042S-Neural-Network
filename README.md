Neural Network — FashionMNIST (Baseline + Grid Search)
=====================================================

Overview
--------
This repository contains a single Jupyter notebook, `Neural Network.ipynb`, which implements a simple feed-forward neural network (multilayer perceptron) trained on the FashionMNIST dataset using PyTorch.

The notebook covers:
- Data loading and preprocessing with torchvision.transforms (normalisation and flattening).
- Train/validation split with deterministic seeding.
- Definition of a MultiLayerPerceptron class (two hidden layers, ReLU activations).
- Training loop with metrics tracking (loss + accuracy) and optional early stopping.
- A small hyperparameter grid search over learning rate, optimiser type, and number of neurons in the first hidden layer.
- Visualisation: training curves, a bar plot of hyperparameter results, confusion matrix, and example images.

Files
-----
- `Neural Network.ipynb` — The main notebook implementing the pipeline.
- `data/` — Expected dataset folder (the notebook uses torchvision's `datasets.FashionMNIST` with `root="./data/train"` and `root="./data/test"`). The notebook will download FashionMNIST if the raw files are missing.

Environment & Dependencies
--------------------------
Minimum dependencies used in the notebook:
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch (PyTorch)
- torchvision

You can install the dependencies with pip. For a quick install (recommended to use a virtualenv/conda env):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

(If you use a CUDA-enabled PyTorch build, install the appropriate `torch` wheel following the instructions at https://pytorch.org.)

How to run
----------
1. Open `Neural Network.ipynb` in Jupyter Notebook / JupyterLab / VS Code and run the cells in order.
2. The notebook will download the FashionMNIST data into `./data/train` and `./data/test` the first time it runs.
3. Training and grid search are run in the notebook; each training run is relatively small (default epochs in the notebook are 10) but will take longer if you run the full grid.

Key notebook knobs
------------------
- `SPLIT_SEED` (cell that performs train/val split) — controls reproducible splitting and DataLoader shuffling.
- `train_model(..., num_epochs=...)` — number of training epochs.
- Hyperparameters defined for grid search:
  - `alphas = [0.0001, 0.001, 0.01, 0.1]`
  - `optims = ["Adam", "SGD", "RMSprop"]`
  - `neurons = [100, 150, 200, 250]`

Reproducibility notes
---------------------
The notebook attempts to be deterministic:
- A global split seed (`SPLIT_SEED`) is used for `random_split` and for seeding DataLoader workers.
- `set_deterministic_seed()` sets PyTorch, numpy and random seeds and configures cuDNN for determinism when CUDA is available.

Caveats:
- Determinism on GPU is not guaranteed for all operations / PyTorch versions. If you need bitwise reproducibility, run on CPU or consult the PyTorch reproducibility docs.

Outputs and visualisations
--------------------------
- Training and validation loss/accuracy printed and plotted.
- A `pandas` DataFrame summarises training curves.
- Grid search results are printed and visualised with a seaborn barplot.
- Confusion matrix and example images from the validation set are displayed.

Suggestions & Next steps
------------------------
- Add command-line or script wrappers to run experiments outside the notebook.
- Save model checkpoints to disk to avoid retraining.
- Add unit tests for helper functions (data splits, seed setting) if you plan to extend this into a package.
- Consider using PyTorch Lightning or Hydra for cleaner experiment management.

