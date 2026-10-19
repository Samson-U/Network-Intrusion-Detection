
# Intelligent Network Intrusion Detection Using Machine Learning

## Overview

This project explores supervised machine learning and unsupervised anomaly detection for identifying network attacks using the CICIDS2017 dataset.

The project is inspired by the Smart India Hackathon problem statement concerning DDoS protection for cloud environments.

## Objectives

- Audit and preprocess network traffic data.
- Explore feature distributions, correlations and class balance.
- Train and evaluate supervised classification models.
- Develop a custom statistical anomaly detector.
- Compare the custom detector with Isolation Forest.
- Analyze detection performance across attack categories.

## Dataset

Dataset: CICIDS2017 DoS subset.

Original dataset information:
https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains benign traffic and several denial-of-service attack categories.

Check the dataset's usage and redistribution terms before sharing copies publicly.

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
Network-Intrusion-Detection/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── src/
│   ├── data_audit.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_sensitivity.py
│   ├── custom_detector.py
│   └── final_comparison.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Pipeline

Run these commands from the project root, in order:

```powershell
python src/data_audit.py
python src/preprocessing.py
python src/eda.py
python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/feature_sensitivity.py
python src/custom_detector.py
python src/final_comparison.py
```

Running preprocessing and preparation again will regenerate processed datasets and splits.

## Preliminary Results

| Model | Accuracy | F1-score |
|---|---:|---:|
| Logistic Regression | 98.28% | 98.35% |
| Decision Tree | 99.53% | 99.58% |
| Random Forest | 99.55% | 99.56% |
| Isolation Forest | 70.24% | 76.78% |

The supervised models use macro-averaged metrics, while Isolation Forest uses binary attack-versus-benign metrics. These scores are not directly equivalent.

The results are exploratory benchmarks on the current dataset split. Further validation is needed to assess generalization and potential dataset-specific artifacts.

## Limitations

- Evaluation uses a benchmark dataset rather than live network traffic.
- Anomaly detection performance varies by attack category.
- Identical feature-pattern grouping does not guarantee independence between related traffic flows.
- The project does not implement live traffic monitoring or automatic network blocking.

## Future Improvements

- Validate on independent captures or suitable grouped splits.
- Investigate DDoS detection errors.
- Evaluate robustness to distribution shifts.
- Explore real-time inference in a controlled test environment.

## Author

Samson Joseph Eappen
