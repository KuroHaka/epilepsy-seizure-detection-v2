# Epilepsy Seizure Detection v2

Automatic epileptic seizure detection from scalp EEG using the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). Five ML/DL models classify 2-second EEG epochs as seizure or non-seizure.

## Models

| Model | Type |
|---|---|
| KNN | k-Nearest Neighbours (k=60) |
| SVM | Support Vector Machine (linear) |
| Random Forest | 100 estimators |
| FFNN | PyTorch fully-connected (102→64→32→16→1) |
| CNN | PyTorch 2D-Conv over frequency features + FFNN over time features |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Clone and enter the repo
git clone https://github.com/KuroHaka/epilepsy-seizure-detection-v2
cd epilepsy-seizure-detection-v2

# Install dependencies (CUDA 12.4 torch is pulled automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate   # Linux/WSL
# .venv\Scripts\activate    # Windows
```

## Dataset

Download the [CHB-MIT dataset](https://physionet.org/content/chbmit/1.0.0/) and update the `drive_path` variable at the top of each script to point to your local copy:

```python
drive_path = "/path/to/chb-mit/"  # update in generate_train_test.py, merge_train_test.py, plotter.py
```

## Workflow

### 1. Generate features

```bash
uv run python generate_train_test.py
uv run python merge_train_test.py
```

This reads `.edf` files, splits into 2-second epochs (1-second overlap), extracts 102 features per epoch (time-domain + FFT), and saves balanced datasets to `data/train.pickle` and `data/test.pickle`.

### 2. Train models

Open and run `model.ipynb`. Trained models are saved to `Models/` as `.pt` files.

### 3. Streamlit app

```bash
uv run streamlit run app.py
```

Pick an EDF file and model in the UI, then click **Autodetect** to visualise ground-truth seizures (green) vs predictions (red).

### 4. CLI inference

```bash
uv run python plotter.py <edf_filename> <model_name>
# e.g.: uv run python plotter.py chb01_03.edf CNN
```

## Notes

- `Models/` is git-ignored — you must train before running inference.
- `standarization_values.json` stores mean/std used for inference-time normalisation.
- The Streamlit app and plotter use `matplotlib Qt5Agg` for interactive MNE windows — requires a display (WSLg or X server on WSL).
