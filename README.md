AI-Based Digital Twin & Hybrid Modeling Platform
================================================

This repository implements a reference architecture for building digital twins and hybrid physics/ML models for intelligent control and predictive maintenance of complex technological processes (e.g., thermal reactors, furnaces, or heat exchangers). It includes:

- Nonlinear simulation of a thermal process with disturbances and noise.
- Data acquisition and preprocessing pipeline with feature extraction (time and frequency domain).
- Hybrid modeling that blends physics-based simulation, residual neural networks (PyTorch LSTM/GRU), and a simple physics-informed neural network (PINN) example.
- Reliability and predictive maintenance simulation for control vs. AI-assisted strategies, with statistical analysis and visualization.
- Reproducible experiment driver that runs end-to-end, produces metrics, plots, and summaries.

## Quick start

```bash
python -m venv venv
venv/Scripts/activate  # Windows
pip install -r requirements.txt
python -m src.experiments.run_experiment --horizon 200 --n_control 20 --n_experiment 20
```

Artifacts are saved under `results/` (CSV/JSON summaries) and `figures/` (plots). PlantUML diagram for the experiment design lives in `plantuml/architecture_experiment.puml`.

## Running tests

```bash
pytest
```

Torch/scipy/sklearn must be installed (see `requirements.txt`).

## Project layout

- `src/simulation/` – nonlinear state-space process model and fault scenarios.
- `src/data_pipeline/` – mocked acquisition, preprocessing, and feature extraction.
- `src/models/` – physics model wrapper, residual PyTorch models, PINN, and ensemble ML utilities.
- `src/maintenance/` – control vs. AI maintenance policies and reliability metrics.
- `src/experiments/` – experiment design and CLI entry point.
- `src/analysis/` – statistical tests and textual reporting.
- `src/visualization/` – plotting utilities.
- `tests/` – smoke tests for simulation and models.
- `figures/`, `results/` – output folders.
- `plantuml/` – experiment architecture diagram source.

## Notes

- Python 3.10+ required.
- Torch is used for neural components; CPU-only installs work for this example.
- XGBoost is optional; install it separately if you want to try gradient boosted trees (`pip install xgboost`). The code falls back gracefully if it is absent.
