# Experiment Approach and Configuration Guide

## 1. Purpose

This project compares two models on a binary handwritten-digit classification
task:

1. A Variational Quantum Classifier (VQC) executed by Qiskit Aer on a
   classical computer.
2. A classical Support Vector Machine (SVM) baseline.

The comparison is intended for demonstration and research discussion. It is not
evidence that quantum models are universally better than classical models.

## 2. What are digit classes?

A **digit class** is the label assigned to an image according to the digit it
represents. In this dataset, the available classes are the ten labels `0`
through `9`. For example, every image of a handwritten `3` belongs to class
`3`, while every image of a handwritten `7` belongs to class `7`.

This application performs **binary classification**, which means it selects
exactly two digit classes and asks the models to decide which of those two
digits appears in an image. With the default settings, the task is:

```text
Input:  an 8×8 handwritten digit image
Output: either class 3 or class 7
```

The class labels remain the original digit values in the displayed results, so
predictions are reported as `3` or `7` rather than as unexplained values such as
`0` and `1`.

### Why must the two digit classes be different?

The VQC and SVM are configured to distinguish two separate categories. If both
sidebar controls selected the same digit, for example `3` versus `3`, there
would be no classification problem: every selected image would have the same
label. This would make precision, recall, confusion matrices, and model
comparisons scientifically meaningless.

The application therefore rejects identical classes. It also restricts both
values to the valid dataset labels `0` through `9`.

### How should class pairs be selected?

Different pairs represent different levels of classification difficulty:

- `0` vs `1` may be visually easier because their shapes are very different.
- `3` vs `8` may be harder because some handwriting styles make them similar.
- `4` vs `9` can contain overlapping shapes depending on the writer.

For a fair comparison, keep the same class pair, preprocessing settings, test
split, and random seed when comparing the VQC with the SVM. To study how the
problem changes, deliberately change only the class pair and record the
resulting metrics. A model that performs well on one pair is not automatically
expected to perform equally well on every other pair.

## 3. End-to-end workflow

The application follows this pipeline:

```text
scikit-learn digits dataset
        ↓
select two digit classes
        ↓
normalize 8×8 pixels to [0, 1]
        ↓
stratified train/test split
        ↓
fit PCA on training data only
        ↓
scale PCA features to [-π, π]
        ↓
train VQC and SVM on the same training split
        ↓
predict the same test split
        ↓
compare metrics, confusion matrices, and training time
```

The PCA transformation is fitted only on the training data. The test data is
transformed using the already-fitted PCA object so that test-set information
does not leak into training.

## 4. User-configurable settings

The controls are located in the Streamlit sidebar.

| Setting | Default | Allowed values | Meaning |
|---|---:|---|---|
| Digit class A | `3` | `0`–`9` | First class in the binary task |
| Digit class B | `7` | `0`–`9` | Second class; must differ from class A |
| PCA features / qubits | `4` | `2`, `4`, `8` | Number of retained PCA features and VQC qubits |
| Training sample limit | `200` | `20`–`500` | Maximum number of selected samples used |
| Test set size | `0.25` | `0.10`–`0.40` | Fraction reserved for testing |
| COBYLA iterations | `20` | `1`–`100` | Maximum VQC optimizer iterations |
| Simulator shots | `256` | `64`, `128`, `256`, `512` | Measurement repetitions per circuit evaluation |
| Random seed | `42` | `0`–`100000` | Controls dataset sampling, splitting, and simulator behavior |

### Recommended starting configuration

Use the defaults for a quick demonstration:

```text
Classes: 3 and 7
PCA features/qubits: 4
Samples: 200
Test size: 25%
COBYLA iterations: 20
Shots: 256
Seed: 42
```

For a faster classroom demonstration, use 2 features, 20–50 optimizer
iterations, and 64 or 128 shots. For a more stable but slower simulation, use
4 or 8 features, more iterations, and 512 shots.

## 5. Dataset and preprocessing interpretation

The default dataset is the offline `sklearn.datasets.load_digits` dataset:

- 8×8 grayscale images
- Pixel values originally range from 0 to 16
- Pixel values are normalized to `[0, 1]`
- Only the two selected classes are retained
- Stratification preserves the class balance in train and test partitions

PCA reduces the original 64-pixel vector to the selected number of components.
Whitening is enabled so the retained components have comparable scale. Each
component is then scaled using training-set extrema and clipped to `[-π, π]`.
These values are used as angles in the quantum feature map.

The selected number of components also determines the number of qubits. This
keeps the feature dimension compatible with the VQC input dimension.

## 6. Quantum model configuration

The VQC uses:

- Qiskit `ZFeatureMap` for data encoding
- Qiskit `TwoLocal` ansatz with `ry` and `rz` rotations
- Linear controlled-Z entanglement
- One ansatz repetition
- COBYLA as the classical optimizer
- Qiskit Aer `SamplerV2` for local simulation
- The configured shot count and random seed

The generated circuit shown in the **Quantum Circuit** tab is the actual
feature-map-plus-ansatz circuit used by the model. Its displayed qubit count,
operation count, and depth describe circuit complexity for the selected
configuration.

### Effect of the quantum settings

- **More qubits/features:** retain more image information, but increase circuit
  size and simulation cost.
- **More COBYLA iterations:** give the optimizer more opportunity to improve
  parameters, but increase training time.
- **More shots:** reduce sampling noise in measured probabilities, but increase
  computation time.
- **Different seed:** changes the sampled subset/split and stochastic
  simulation details. Do not compare runs with different seeds as if they were
  identical experiments.

COBYLA requires at least `number_of_trainable_parameters + 2` function
evaluations. The application automatically raises a smaller requested
iteration budget to that minimum so SciPy does not emit an invalid `MAXFUN`
warning. Qiskit may also print informational messages about creating a default
gradient and transpilation; these are not failures and the VQC still uses the
configured Aer sampler.

## 7. Classical baseline

The SVM is trained on the same PCA-transformed training features as the VQC and
evaluated on the same test features. This makes the comparison more controlled:
the main difference is the classifier, not the data split or preprocessing.

The SVM uses an RBF kernel, standard scaling, and probability estimates for
classification. Its training time is measured independently from VQC training.

## 8. How to analyse an experiment

### Step 1: Check the experiment definition

Record the classes, sample limit, test size, PCA dimension, optimizer
iterations, shots, and seed. The experiment JSON download contains these
settings.

### Step 2: Check the dataset

In **Dataset Explorer**, verify:

- Both requested classes are present.
- The sample count is reasonable.
- The displayed images look like handwritten digits.
- PCA feature values are finite and vary across samples.

### Step 3: Compare the main metrics

In **Evaluation**, compare:

- **Accuracy:** fraction of all test samples classified correctly.
- **Precision:** among samples predicted as the positive class, how many were
  correct.
- **Recall:** among actual positive-class samples, how many were found.
- **F1 score:** harmonic mean of precision and recall.
- **Training time:** elapsed model-fitting time in seconds.

Do not focus on accuracy alone if the classes are imbalanced or the two error
types have different importance.

### Step 4: Inspect confusion matrices

The confusion matrix contains:

```text
                 predicted A   predicted B
actual A              true A       A → B errors
actual B              B → A errors true B
```

Look for whether one class is consistently confused with the other. Similar
error patterns between models suggest that the preprocessing or class pair may
be the dominant difficulty.

### Step 5: Inspect the classification reports

The report provides per-class precision, recall, and F1 values. Check whether a
model performs well on both classes or only on one class. A small difference in
overall accuracy may hide a meaningful difference for a particular digit.

### Step 6: Interpret training time and circuit complexity

Relate training time to:

- Number of samples
- Number of PCA features/qubits
- COBYLA iterations
- Simulator shots
- Circuit depth and operation count

Simulation time is not equivalent to execution time on future quantum
hardware. It includes classical simulation overhead and should be reported as
an engineering measurement, not a hardware performance claim.

## 9. Suggested comparison experiments

For a poster or report, keep the seed fixed while changing one factor at a
time.

### Experiment A: Feature dimension

Run with 2, 4, and 8 PCA features. Compare accuracy, F1 score, training time,
and circuit depth. This shows the trade-off between dimensionality reduction
and circuit complexity.

### Experiment B: Optimizer budget

Keep all other settings fixed and compare 5, 20, and 50 COBYLA iterations.
Check whether additional iterations improve test performance enough to justify
the extra time.

### Experiment C: Measurement noise

Keep the model and seed fixed and compare 64, 256, and 512 shots. Look for
changes in VQC predictions and metrics caused by finite sampling.

### Experiment D: Class-pair difficulty

Compare visually different pairs, such as `0` vs `1`, with more similar pairs,
such as `3` vs `8`. Do not compare only the best-performing pair.

### Experiment E: Repeated seeds

Repeat a selected configuration with several seeds. Report the mean and spread
of metrics rather than relying on one favorable split. The current UI reports
one run at a time, so record each downloaded metrics CSV and experiment JSON.

## 10. Important limitations

- The dataset is small and contains only 8×8 images.
- Only binary classification is implemented.
- PCA discards information before classification.
- The VQC uses a small circuit and a limited optimizer budget.
- Aer simulation is not execution on physical quantum hardware.
- One train/test split can produce unstable conclusions.
- Training time depends on the local machine and should not be generalized.
- A higher VQC score on one run does not demonstrate quantum advantage.

## 11. Reproducibility checklist

Before reporting results, save:

1. The downloaded experiment JSON.
2. The downloaded metrics CSV.
3. The downloaded predictions CSV.
4. The selected digit pair.
5. The Python and package versions.
6. The machine/environment used for simulation.
7. Any warnings or failed runs.

Use the same dependency versions from `requirements.txt`, keep the random seed
visible, and rerun the tests with:

```powershell
python -m pytest
```

Launch the application with:

```powershell
python -m streamlit run app.py
```
