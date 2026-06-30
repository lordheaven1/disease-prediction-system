# Disease Prediction System

A machine learning pipeline that predicts disease from patient-reported symptoms. The project trains and benchmarks five classifiers, investigates why a 100% accuracy score on clean synthetic data is a red flag rather than a win, stress-tests the models against injected real-world noise, tunes the most robust model via GridSearchCV, and deploys the final model through a desktop GUI.

## Overview

Given a set of patient-reported symptoms, the system predicts the most likely disease from a dataset covering **41 diseases** and **131 symptoms**. The project walks through the full ML lifecycle: data cleaning, model benchmarking, robustness testing, hyperparameter tuning, and deployment.

## Key Highlights

- **Five classifiers benchmarked**: Decision Tree, Random Forest, SVM, XGBoost, and MLP
- **Caught a red flag early**: initial models hit 100% accuracy on the clean dataset, a sign of an unrealistic/overfit setup rather than genuine model quality
- **Synthetic noise injection**: added controlled noise to simulate real-world, imperfect symptom reporting and re-evaluated all models under more realistic conditions
- **Hyperparameter tuning** via `GridSearchCV` to optimize the most promising models
- **Final model deployment**: the best-performing model (SVM) was serialized with `joblib` and wrapped in a Tkinter desktop GUI for live predictions

## Results

After noise injection and tuning, final model rankings by accuracy:

| Model | Accuracy |
|---|---|
| SVM | 85.22% |
| XGBoost | 84.78% |
| Random Forest | 84.18% |
| Decision Tree | 49.85% |

SVM was selected as the final model for deployment based on its accuracy and robustness to noisy input.

## Tech Stack

- **Language**: Python
- **ML/Data**: scikit-learn, XGBoost, pandas, NumPy
- **Model tuning**: GridSearchCV
- **Model persistence**: joblib
- **GUI**: Tkinter
- **Development**: Jupyter Notebook

## How It Works

1. **Data cleaning** — preprocessing the symptom-disease dataset to remove inconsistencies
2. **Baseline training** — training all five classifiers on the clean dataset
3. **Red flag detection** — identifying suspiciously perfect accuracy scores and diagnosing the cause
4. **Noise injection** — introducing synthetic noise to simulate realistic, imperfect symptom input
5. **Re-evaluation** — re-running all models against the noisy dataset for a fair, robust comparison
6. **Hyperparameter tuning** — using GridSearchCV to optimize the best-performing model (SVM)
7. **Deployment** — serializing the final pipeline and building a simple Tkinter GUI for 3-symptom input and live disease prediction

## Project Structure

```
disease-prediction-system/
├── Project.ipynb                  # Main notebook: data prep, training, evaluation, tuning
├── dataset.csv                    # Symptom-disease training dataset
├── Symptom-severity.csv           # Symptom severity weights
├── symptom_Description.csv        # Disease descriptions
├── symptom_precaution.csv         # Precautions per disease
└── README.md
```

## Running the Project

1. Clone the repository:
   ```
   git clone https://github.com/lordheaven1/disease-prediction-system.git
   cd disease-prediction-system
   ```
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn xgboost joblib
   ```
3. Open `Project.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order.

## Notes

This project was built as part of academic coursework (CSE3968, Machine Learning) and emphasizes not just model accuracy but model trustworthiness, recognizing and correcting for unrealistic results being a core part of the workflow.

## Author

**D N Sahil** — [GitHub](https://github.com/lordheaven1)
