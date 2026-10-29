# Quantum Machine Learning for Image Classification

An offline Streamlit research demonstration comparing a Qiskit Aer-simulated
Variational Quantum Classifier (VQC) with an SVM on two classes from the
scikit-learn 8×8 handwritten digits dataset.

## Setup

Python 3.11 or 3.12 is recommended:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run tests with `python -m pytest`.

## Architecture

`src/data_loader.py` loads and stratifies the offline digits data.
`src/preprocessing.py` fits PCA on the training partition only and scales
features to rotation angles. `src/quantum_model.py` builds a Z feature map,
TwoLocal ansatz, COBYLA optimizer, and Aer `SamplerV2`-backed Qiskit ML VQC.
`src/classical_model.py` trains the same split with an SVM. Evaluation,
visualization, and Streamlit presentation are separate modules.

## Reproducibility and interpretation

The UI records classes, PCA size, split, seed, optimizer iterations, and shots.
Metrics and predictions are generated from each run and can be downloaded.
Simulation is not physical hardware, and a small dataset/circuit cannot
establish quantum advantage. The classical model is a baseline, not a control
for a claim of superiority.

If Qiskit primitives fail after an environment upgrade, recreate the virtual
environment and reinstall the pinned requirements. Qiskit packages should be
installed as a compatible set rather than upgraded independently.
