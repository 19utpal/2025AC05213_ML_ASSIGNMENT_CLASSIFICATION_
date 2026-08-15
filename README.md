# SDSS Stellar Classification

## a. Problem Statement

This project classifies Sloan Digital Sky Survey astronomical objects into three target classes: GALAXY, STAR, and QSO. The Streamlit application trains multiple classification models on the same embedded SDSS dataset and compares their performance using standard evaluation metrics.

The app is designed as a self-contained `app.py` file. The reference dataset and model logic are embedded directly in the Python file, so the app does not require a separate CSV file, saved model file, or scikit-learn installation.

## b. Dataset Description

The dataset contains 1,000 SDSS astronomical records. Each record includes celestial coordinates, photometric magnitude bands, redshift, engineered color-index features, and a class label.

Input features used for training and prediction:

- `alpha`
- `delta`
- `u`
- `g`
- `r`
- `i`
- `z`
- `redshift`
- `u_g_color`
- `g_r_color`
- `r_i_color`
- `i_z_color`

Target column:

- `class`: GALAXY, STAR, or QSO

The app also supports uploading a new CSV file with the required feature columns. If the uploaded file includes a `class` column, the app calculates evaluation metrics for that uploaded data as well.

## c. GitHub Repository Link

Add your GitHub repository link here before final submission:

```text
[https://github.com/<your-username>/<your-repository-name>](https://github.com/19utpal/2025AC05213_ML_ASSIGNMENT_CLASSIFICATION_.git)
```

## d. Models Used

The following six classification models are implemented in `app.py` and evaluated on the same dataset:

| ML Model Name | Description |
| --- | --- |
| Logistic Regression | Softmax logistic regression implemented with NumPy gradient descent |
| Decision Tree | Custom Gini-based decision tree classifier |
| kNN | Custom K-Nearest Neighbor classifier |
| Naive Bayes | Custom Gaussian Naive Bayes classifier |
| Random Forest (Ensemble) | Custom ensemble of bootstrapped decision trees |
| Gradient Boosting | Custom boosted tree-style ensemble |

The app calculates the following metrics for every model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC Score)

The full comparison table is displayed in the Streamlit app under the **Model Comparison** tab.

## Model Comparison Table

The Streamlit app generates this table dynamically at runtime:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |
| Decision Tree | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |
| kNN | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |
| Naive Bayes | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |
| Random Forest (Ensemble) | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |
| Gradient Boosting | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app | Calculated in app |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline; works well when scaled photometric features separate classes with near-linear boundaries. |
| Decision Tree | Captures non-linear rules but may overfit individual splits, making it useful as a comparison against ensemble models. |
| kNN | Distance-based model that benefits from scaling; performance depends on local neighborhood structure. |
| Naive Bayes | Fast probabilistic baseline; can be weaker because magnitude and color-band features are correlated. |
| Random Forest (Ensemble) | Combines many trees to reduce overfitting and usually gives stable performance on tabular astronomy features. |
| Gradient Boosting | Sequential ensemble that can model non-linear feature interactions and often competes for the best score. |
| Overall Winner | Displayed dynamically in the app based on F1 Score, MCC Score, and Accuracy on the holdout split. |

## How to Run

Install the runtime packages available in the environment:

```bash
pip install streamlit numpy pandas
```

Run the app:

```bash
streamlit run app.py
```

## Deployment Notes

This version intentionally avoids external project-file dependencies:

- No external `test_csv.csv` is required.
- No saved model files are required.
- No `scikit-learn` package is required.
- The dataset is embedded directly in `app.py`.
- The classification models and metrics are implemented directly in `app.py`.

For the BITS Virtual Lab screenshot requirement, open the app and capture the **Model Comparison** tab, because it shows all implemented models and their evaluation metrics.
