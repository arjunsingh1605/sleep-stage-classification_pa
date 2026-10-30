# Sleep Stage Classification Using Machine Learning

A machine-learning project that classifies 30-second polysomnography (PSG) epochs into five sleep stages using engineered signal features and Logistic Regression.

## Overview

Sleep Stage Classification (SSCP) explores how physiological signals recorded during sleep can be converted into features that help a machine-learning model estimate a person's sleep stage.

Sleep is not one uniform state. Across the night, the brain moves through Wake, non-rapid eye movement (NREM) stages N1, N2 and N3, and rapid eye movement (REM) sleep. These stages differ in characteristic brain activity, eye movements, muscle activity and patterns over time.

Manually scoring sleep stages involves examining physiological recordings and assigning a stage to each short epoch. This project investigates a reproducible machine-learning workflow that extracts numerical features from PSG recordings and predicts one of five stage labels.

**Current scope:** an offline, epoch-level classification baseline evaluated on held-out subjects. It is a research and learning prototype, not a clinical diagnostic system or a real-time sleep monitor.

## Why Sleep Stage Classification?

Sleep-stage estimates help researchers and clinicians describe sleep architecture: how sleep is distributed across a night, how often stages change, and how much time is spent in different stages.

Automating parts of this process may help with:

- **Sleep research:** analyzing large collections of overnight recordings consistently.
- **Clinical workflow support:** assisting trained sleep specialists by prioritizing or pre-scoring epochs for review. Human review remains essential.
- **Sleep disorder research:** studying sleep architecture in conditions such as insomnia, sleep apnea and narcolepsy. Stage classification alone does not diagnose these conditions.
- **Wearable and home monitoring research:** exploring whether physiological signals from less intrusive devices can support sleep-stage estimates.
- **Longitudinal analysis:** comparing sleep patterns across nights or groups in research settings.
- **Education and prototyping:** demonstrating signal processing, feature engineering, imbalanced classification and subject-independent evaluation.

These are potential application areas for the broader field. This project has not been clinically validated and does not establish that its predictions are suitable for treatment decisions or consumer sleep tracking.

## Dataset

The project uses the **Sleep-EDF Expanded Sleep Cassette** recordings from PhysioNet. PSG recordings are paired with hypnogram annotations that provide reference sleep-stage labels.

The current processed dataset contains:

| Dataset property | Value |
|---|---:|
| Labeled epochs | 414,961 |
| Unique subjects | 78 |
| Recordings | 153 |
| Epoch duration | 30 seconds |
| Sleep-stage classes | 5 |
| Numeric features in V2 | 38 |
| Total columns in V2 | 44 |

Each row represents one labeled 30-second epoch. The V2 dataset has 44 columns: 38 numerical features and six metadata/label columns.

### Sleep stages

| Label | Meaning |
|---|---|
| `W` | Wake |
| `N1` | NREM stage 1, a transition into sleep |
| `N2` | NREM stage 2 |
| `N3` | Deep NREM sleep |
| `REM` | Rapid eye movement sleep |

The source annotations' stages 3 and 4 are combined into `N3` for the five-class task. Unscored or unsupported labels are excluded during epoch construction.

### Class imbalance

The dataset is strongly imbalanced:

| Stage | Epoch count | Proportion |
|---|---:|---:|
| W | 285,433 | 68.79% |
| N1 | 21,522 | 5.19% |
| N2 | 69,132 | 16.66% |
| N3 | 13,039 | 3.14% |
| REM | 25,835 | 6.23% |

Wake dominates the dataset. As a result, accuracy alone can make a model appear stronger than its performance across all stages actually is. Balanced accuracy, macro F1 and per-stage recall are therefore important evaluation metrics.

## Project Workflow

1. **Load recordings and annotations**  
   Read PSG signals and their matching hypnogram files using MNE-based data-loading utilities.

2. **Construct labeled epochs**  
   Divide recordings into 30-second epochs and map source annotations to `W`, `N1`, `N2`, `N3` and `REM`.

3. **Audit the dataset**  
   Check class counts, subject and recording identifiers, missing values, infinite values and duplicate recording/epoch keys.

4. **Extract numerical features**  
   Compute statistical and frequency-domain features from EEG and selected physiological channels.

5. **Explore the features**  
   Compare feature distributions across stages and inspect whether signal characteristics show useful separation.

6. **Create a subject-independent split**  
   Use `GroupShuffleSplit` with `subject_id` as the grouping variable. With `test_size=0.20` and `random_state=42`, 62 subjects are used for training and 16 for testing. No subject appears in both partitions, reducing the risk of leakage between nights from the same person.

7. **Train a dummy baseline**  
   Fit a most-frequent classifier to establish a simple reference point.

8. **Train Logistic Regression**  
   Standardize numerical features and fit a multiclass Logistic Regression pipeline. Compare unweighted and class-weighted variants.

9. **Evaluate and visualize**  
   Measure accuracy, balanced accuracy, macro F1, per-stage precision/recall/F1 and confusion matrices. Display epoch-level predictions and compare actual and predicted stages over a recording timeline.

## Feature Engineering

The feature pipeline currently uses 38 numeric features in V2.

### Statistical features
For selected signals, calculate summary statistics such as mean, standard deviation, root mean square (RMS) and peak-to-peak amplitude.

### Frequency-domain features
Use Welch's power spectral density estimate to calculate EEG power in five frequency bands:

| Band | Frequency range |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Sigma | 12–16 Hz |
| Beta | 16–30 Hz |

Both absolute band power and relative band power are included for the two EEG channels used by the feature extractor.

### Additional V2 spectral features
V2 includes six additional features across the two EEG channels:

- Theta-to-alpha power ratio
- Delta-to-beta power ratio
- Spectral entropy calculated from the integrated band powers

The six additional features were evaluated as a feature-engineering experiment. Their contribution was modest rather than transformative.

## Model and Results

### Selected baseline

The current baseline is a **StandardScaler + Logistic Regression** pipeline using the V2 feature set. The recorded evaluation used a subject-level train/test split.

| Metric | V2 result |
|---|---:|
| Accuracy | 88.58% |
| Balanced accuracy | 66.94% |
| Macro F1 | 68.23% |

Small differences from earlier recorded runs can occur if the dataset, row order or split differs. These numbers describe the current saved-dataset run.

### Per-stage performance

The following values are from the earlier recorded V2 classification report:

| Stage | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Wake (W) | 0.96 | 0.97 | 0.97 |
| N1 | 0.28 | 0.15 | 0.20 |
| N2 | 0.79 | 0.87 | 0.83 |
| N3 | 0.89 | 0.70 | 0.79 |
| REM | 0.62 | 0.65 | 0.63 |

The current saved-dataset run reported accuracy 0.8858, balanced accuracy 0.6694 and macro F1 0.6823. Re-run the classification report if exact per-stage metrics corresponding to that run are needed.

### Baseline comparison

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Dummy most-frequent classifier | 0.6890 | 0.2000 | 0.1632 |
| Logistic Regression V1 | 0.8781 | 0.6551 | 0.6647 |
| Class-weighted Logistic Regression V1 | 0.7980 | 0.7180 | 0.6327 |
| Logistic Regression V2 | 0.8858 | 0.6694 | 0.6823 |

The V2 feature set modestly improved the recorded accuracy and macro F1 over V1. Class weighting improved balanced accuracy in the earlier experiment but reduced accuracy and macro F1, illustrating the trade-off between minority-class recall and overall performance.

### Interpreting the results

- **Wake** is classified well, but its large share of the dataset contributes substantially to overall accuracy.
- **N2** has relatively strong recall and F1.
- **N3** has high precision but lower recall than Wake and N2.
- **REM** is moderately recognized and is sometimes confused with other stages.
- **N1** remains the weakest class. Its low recall indicates that many actual N1 epochs are assigned to another stage, particularly N2, Wake or REM.

A dummy model that predicts Wake for every epoch reaches about 68.9% accuracy while achieving only 20% balanced accuracy. This is why the project reports multiple metrics instead of presenting accuracy in isolation.

## Visualizations

The baseline notebook includes:

- Class distribution and feature-distribution plots
- A classification report
- A confusion matrix, including a normalized version
- A table of actual versus predicted labels for test epochs
- An actual-versus-predicted sleep-stage timeline for a selected test recording
- A zoomed view of the sleep period

The timeline is a visualization of offline predictions on held-out data. It is not evidence of real-time operation. Recording-level accuracy may also be misleading when Wake dominates the recording, so it should be interpreted alongside per-stage and balanced metrics.

## Repository Structure

```text
sscp/
├── data/
│   ├── raw/
│   │   ├── sleep-cassette/
│   │   └── sleep-telemetry/
│   └── processed/
│       ├── sleep_epochs_corrected.csv
│       ├── sleep_epochs_v2.csv
│       └── recording_features_v2/
├── models/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_temporal_prediction.ipynb
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   └── preprocessing.py
├── results/
│   └── baseline_v2/
│       ├── test_predictions.csv
│       ├── classification_report.csv
│       ├── summary_metrics.csv
│       └── confusion_matrix.png
├── README.md
└── requirements.txt
```

The `results/baseline_v2/` files are created by the results-export cell in the baseline notebook. They may not exist until that cell has been run. The notebooks for model comparison and temporal prediction are planned project components; their presence in the repository does not imply those experiments are complete.

## Getting Started

### Requirements

- Python 3.10 or a compatible version supported by the packages in `requirements.txt`
- The Sleep-EDF Expanded Sleep Cassette data files
- Python packages listed in `requirements.txt`

### Setup

```bash
git clone <repository-url>
cd sscp

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the downloaded Sleep-EDF files in the expected directories under `data/raw/`. Follow the dataset's terms of use and preserve the original file names expected by the loader.

### Run the baseline notebook

Open the project in Jupyter and execute `notebooks/04_baseline_model.ipynb` in order. The notebook loads the processed V2 CSV, constructs a subject-level split, trains the Logistic Regression pipeline, evaluates it, and generates prediction visualizations.

The exact paths assume the notebook is run from the project's `notebooks/` directory. If you run cells from another working directory, adjust `PROJECT_ROOT` or the dataset path accordingly.

## Limitations

1. **Class imbalance:** Wake represents nearly 69% of labeled epochs, and N1 is poorly recalled.
2. **Feature limitations:** 30-second summary statistics and band powers can hide brief events and may not separate similar stages.
3. **Independent epoch predictions:** The baseline does not model temporal transitions between neighbouring epochs.
4. **Dataset generalization:** Evaluation uses a subject-level split within Sleep-EDF. Performance on other datasets, devices, populations or clinical settings is unknown.
5. **No clinical validation:** The model is an educational/research prototype and should not be used to diagnose or treat a sleep disorder.
6. **Experimental spindle detector:** A sigma-burst candidate detector was explored, but it has not been validated against expert-labelled spindle events and is not part of the reported baseline.
7. **Reproducibility:** Results depend on the exact processed dataset, feature schema, row order, software versions and split. Preserve these details when reproducing the metrics.

## Future Additions

1. **Improve N1 recognition**
   - Inspect N1 errors and confusion patterns.
   - Test class weighting or resampling carefully.
   - Evaluate richer features without relying on accuracy alone.

2. **Compare additional models**
   - Random Forest and Extra Trees
   - Gradient-boosting methods
   - Other models where justified by validation results

3. **Add multimodal features**
   - Investigate EOG and available submental EMG signals.
   - Check sampling rates, channel availability and signal quality before feature extraction.

4. **Explore temporal context**
   - Use neighbouring epochs or a sequence-based approach.
   - Compare against the independent-epoch baseline with subject-level evaluation preserved.
   - Avoid using future labels or information unavailable at prediction time.

5. **Validate event-based features**
   - Improve spindle or other transient-event features only with a defensible validation strategy, ideally using expert-labelled events.

6. **Strengthen evaluation**
   - Use additional subject-level splits or cross-validation.
   - Report confidence intervals and per-stage metrics.
   - Test generalization on an independent dataset if available.

7. **Build a demonstration interface**
   - Allow a user to load a supported recording and view predicted stages, confidence scores and a timeline.
   - Clearly label outputs as model estimates, retain quality checks and avoid presenting them as diagnoses.

8. **Improve reproducibility and packaging**
   - Save model artifacts and configuration.
   - Record package versions and split identifiers.
   - Add automated tests for epoch construction, feature integrity and prediction output.

## Ethical and Safety Note

Sleep-stage predictions can be wrong, particularly for underrepresented or difficult-to-distinguish stages. This project is intended for learning and research. It is not a substitute for professional sleep-study interpretation, medical advice or diagnosis.

## Acknowledgements

- Sleep-EDF Expanded dataset, distributed through PhysioNet.
- Python ecosystem, including MNE, NumPy, pandas, SciPy, scikit-learn and Matplotlib.

## License and Data

Add the project's chosen software license before redistributing this repository. Review and follow the dataset's own access and usage terms; a software license does not automatically grant rights to redistribute the data.