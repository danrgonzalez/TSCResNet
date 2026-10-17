# TSCResNet

A one-dimensional residual convolutional network for time-series classification, applied to human activity recognition (HAR) from wearable sensors. This is the code from my doctoral research at Colorado Technical University, covering both closed-set classification (every activity is seen during training) and open-set classification (the model must also reject activities it has never seen).

## Model

- **Input:** windows of 40 sensor channels × 172 time steps (5.12 s at 33.3 Hz) from the [PAMAP2](https://doi.org/10.24432/C5NW2H) dataset.
- **Architecture:** strided 1-D convolutions and bottleneck residual blocks (a width-1 convolution that halves the channels, a width-3 convolution that restores them, plus the skip connection), each with batch normalization and LeakyReLU. The layer stack is declared in [`app/architecture.py`](app/architecture.py) and built in [`app/model.py`](app/model.py).
- **No pooling or dense head:** the final convolutions shrink the time axis to 18 positions, which serve directly as the 18 class logits. The model therefore expects 172-step windows.
- **Open set:** one model is trained per activity with that activity held out; at evaluation time, windows whose maximum softmax confidence falls below a threshold are labeled as the unseen activity.

## Results

On the standard PAMAP2 protocol (subject 105 for validation, subject 106 for testing), the closed-set model reached **95.60% accuracy, 95.49% weighted F1, and 95.50% mean F1**, as reported in the 2022 paper below. The open-set results are in the 2023 paper.

## Data

[PAMAP2 Physical Activity Monitoring](https://doi.org/10.24432/C5NW2H) (UCI, CC BY 4.0). Preprocessing follows the papers: resample to 33.3 Hz, fill gaps by linear interpolation, drop transient (label 0) data and the invalid orientation columns (leaving 40 channels), and cut 5.12 s windows with 78% overlap. The preprocessing notebooks are not part of this repository.

The loaders in [`app/dataset.py`](app/dataset.py) expect preprocessed, flattened windows as CSV files:

```
data/
├── train/  X_train.csv, y_train.csv            (open set: X_train_<activity>.csv, y_train_<activity>.csv)
├── val/    X_val.csv,   y_val.csv              (open set: X_val_open_set.csv,  y_val_open_set.csv)
└── test/   X_test.csv,  y_test.csv             (open set: X_test_open_set.csv, y_test_open_set.csv)
```

Each `X_*.csv` row is one window of 40 × 172 values; each `y_*.csv` row is a one-hot vector over the 18 PAMAP2 activities in [`app/activity_to_encoding_map.txt`](app/activity_to_encoding_map.txt). The data directory defaults to `data/` at the repository root; set `TSC_DATA_DIR` to use another location. Checkpoints (`models/`) and MLflow runs (`mlruns/`) are written under the same directory.

## Setup

```bash
conda env create -f environment.yml
conda activate TSCResNet
```

`environment.yml` lists the direct dependencies (Python 3.8, PyTorch, scikit-learn, MLflow, pytest). `environment_20220207.yml` is the fully pinned environment used for the published experiments.

## Train

Hyperparameters and the closed-/open-set switch (`OPEN_SET`) are in [`app/config.py`](app/config.py). Training was run with Adam (learning rate 1e-4), batch size 16, and 50 epochs.

```bash
cd app
python train.py
```

Runs are tracked with MLflow (hyperparameters, training loss, and per-epoch validation and test accuracy and F1):

```bash
mlflow ui --backend-store-uri data/mlruns
```

## Test

```bash
python -m pytest
```

Or in Docker, as CI does:

```bash
docker compose build test
docker compose run --rm test
```

## Citation

```bibtex
@article{rodriguezgonzalez2022closed,
  author  = {Rodriguez Gonzalez, Daniel and Qu, Yanzhen},
  title   = {Improving the Performance of Closed-Set Classification in Human
             Activity Recognition by Applying a Residual Neural Network Architecture},
  journal = {European Journal of Electrical Engineering and Computer Science},
  volume  = {6},
  number  = {2},
  pages   = {59--65},
  year    = {2022},
  doi     = {10.24018/ejece.2022.6.2.433}
}

@inproceedings{rodriguezgonzalez2023openset,
  author    = {Rodriguez Gonzalez, Daniel and Qu, Yanzhen},
  title     = {Performance Evaluation of Open-Set Classification in Human Activity
               Recognition via a Residual Neural Network Architecture},
  booktitle = {2023 Congress in Computer Science, Computer Engineering,
               \& Applied Computing (CSCE)},
  pages     = {470--476},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/CSCE60160.2023.00084}
}
```

## License

[MIT](LICENSE)
