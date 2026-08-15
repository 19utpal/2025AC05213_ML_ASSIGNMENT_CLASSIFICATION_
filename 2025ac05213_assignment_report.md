# Machine Learning Classification Assignment Report

## Student Details

| Field | Details |
| --- | --- |
| Name | Utpal Singh |
| Student ID | 2025AC05213 |
| Project Title | SDSS Stellar Classification |
| Streamlit App Link | https://2025ac05213mlassignmentclassification-jhvpvju4zuscwdcx3etxz6.streamlit.app/ |
| GitHub Repository Link | https://github.com/19utpal/2025AC05213_ML_ASSIGNMENT_CLASSIFICATION_.git |

## 1. Problem Statement

The objective of this assignment is to build a machine learning classification application that classifies Sloan Digital Sky Survey astronomical objects into one of three categories: GALAXY, STAR, or QSO. The project compares multiple classification models on the same dataset and evaluates their performance using standard classification metrics.

The final application is implemented as a Streamlit web app. The dataset, feature engineering, model training, prediction logic, evaluation metrics, and report notes are included in `app.py` so that the application can run without depending on external CSV files or saved model files.

## 2. Dataset Description

The selected dataset is based on SDSS stellar object classification data. It contains 1,000 astronomical records with celestial coordinates, photometric magnitude bands, redshift, engineered color-index features, and the final target class.

### Features Used

| Feature | Description |
| --- | --- |
| `alpha` | Right ascension coordinate |
| `delta` | Declination coordinate |
| `u` | Ultraviolet photometric band magnitude |
| `g` | Green photometric band magnitude |
| `r` | Red photometric band magnitude |
| `i` | Near-infrared photometric band magnitude |
| `z` | Infrared photometric band magnitude |
| `redshift` | Redshift measurement of the astronomical object |
| `u_g_color` | Engineered color feature: `u - g` |
| `g_r_color` | Engineered color feature: `g - r` |
| `r_i_color` | Engineered color feature: `r - i` |
| `i_z_color` | Engineered color feature: `i - z` |

### Target Variable

| Target Column | Classes |
| --- | --- |
| `class` | GALAXY, STAR, QSO |

### Class Distribution

| Class | Number of Records |
| --- | ---: |
| GALAXY | 596 |
| STAR | 209 |
| QSO | 195 |

## 3. Application and Repository Links

| Item | Link |
| --- | --- |
| Streamlit App | https://2025ac05213mlassignmentclassification-jhvpvju4zuscwdcx3etxz6.streamlit.app/ |
| GitHub Repository | https://github.com/19utpal/2025AC05213_ML_ASSIGNMENT_CLASSIFICATION_.git |

## 4. Models Implemented

The following six classification models are implemented and evaluated on the same dataset:

| ML Model Name | Implementation Summary |
| --- | --- |
| Logistic Regression | Softmax logistic regression using NumPy gradient descent |
| Decision Tree | Gini impurity based decision tree classifier |
| kNN | K-Nearest Neighbor classifier using Euclidean distance |
| Naive Bayes | Gaussian Naive Bayes classifier |
| Random Forest (Ensemble) | Ensemble of bootstrapped decision trees |
| Gradient Boosting | Boosted tree-style ensemble model |

The application avoids a `scikit-learn` dependency by implementing the required model logic directly in `app.py` using NumPy and pandas.

## 5. Evaluation Metrics

Each model is evaluated using the following metrics:

| Metric | Meaning |
| --- | --- |
| Accuracy | Overall proportion of correct predictions |
| AUC Score | Weighted one-vs-rest area under the ROC curve |
| Precision | Weighted precision across all classes |
| Recall | Weighted recall across all classes |
| F1 Score | Weighted harmonic mean of precision and recall |
| MCC Score | Matthews Correlation Coefficient for multiclass classification |

The dataset is split using a stratified 75/25 train-test split. Metrics below are calculated on the holdout test split.

## 6. Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.9800 | 0.9881 | 0.9806 | 0.9800 | 0.9799 | 0.9646 |
| Random Forest (Ensemble) | 0.9800 | 0.9919 | 0.9806 | 0.9800 | 0.9798 | 0.9647 |
| Gradient Boosting | 0.9600 | 0.9909 | 0.9623 | 0.9600 | 0.9606 | 0.9304 |
| Naive Bayes | 0.9280 | 0.9833 | 0.9347 | 0.9280 | 0.9290 | 0.8755 |
| Logistic Regression | 0.8680 | 0.9406 | 0.8679 | 0.8680 | 0.8605 | 0.7617 |
| kNN | 0.8560 | 0.9153 | 0.8535 | 0.8560 | 0.8457 | 0.7395 |

## 7. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Logistic Regression provides a useful linear baseline. It performs reasonably well after feature scaling, but its scores are lower than tree-based models because SDSS object classes have non-linear relationships across color bands and redshift. |
| Decision Tree | Decision Tree gives the best F1 Score and very high Accuracy. It captures non-linear splits in the photometric and redshift features effectively, making it the best model by the report's combined ranking rule. |
| kNN | kNN benefits from scaled features but has the lowest performance in this run. Its distance-based behavior is sensitive to local feature spacing and class overlap. |
| Naive Bayes | Naive Bayes performs better than Logistic Regression and kNN. However, it assumes feature independence, which is not fully true for photometric bands and engineered color features. |
| Random Forest (Ensemble) | Random Forest has the highest MCC Score and AUC Score and ties Decision Tree on Accuracy. It is more stable than a single tree because it averages multiple bootstrapped trees. |
| Gradient Boosting | Gradient Boosting performs strongly, especially on AUC and F1 Score. It captures non-linear patterns through sequential boosted trees. |
| Overall Winner for Dataset | Decision Tree is selected as the overall winner because it has the highest F1 Score while also tying for the highest Accuracy. Random Forest is a very close competitor with slightly higher MCC and AUC. |

## 8. Streamlit Application Features

The Streamlit app includes the following screens and functionality:

- Overview tab with dataset size, feature count, class count, metrics, confusion matrix, and predicted class mix.
- Model Comparison tab with the full evaluation table for all six models.
- Predictions tab with classified records and downloadable prediction results.
- Data tab with embedded data preview and feature summary.
- README Notes tab with problem statement, dataset description, model list, dependencies, and GitHub link reminder.
- Sidebar control to select the classification model used for predictions.
- File upload option for another CSV with the required columns.

## 9. Repository Structure

The submitted repository is designed to support a self-contained Streamlit app.

```text
project-folder/
|-- app.py
|-- README.md
|-- ASSIGNMENT_REPORT.md
```

The `app.py` file contains:

- Embedded dataset
- Feature engineering logic
- Classification model implementations
- Evaluation metric implementations
- Streamlit user interface
- Prediction download functionality

## 10. Requirements

The app does not require `scikit-learn`. The required runtime packages are:

```text
streamlit
numpy
pandas
```

If running locally, install them with:

```bash
pip install streamlit numpy pandas
```

Then run:

```bash
streamlit run app.py
```

## 11. BITS Virtual Lab Proof

As required by the assignment, the project should be executed on BITS Virtual Lab and one screenshot should be uploaded as proof.

Recommended screenshot:

- Open the deployed or local Streamlit app.
- Go to the **Model Comparison** tab.
- Capture the table showing all six models and their metrics.

## 12. Conclusion

This assignment successfully implements six classification models on the same SDSS dataset and compares them using Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC Score. The Decision Tree model is selected as the overall winner based on the highest F1 Score and tied highest Accuracy, while Random Forest also performs extremely well with the highest MCC and AUC Score.

The application is deployed on Streamlit and maintained in GitHub using the links below:

- Streamlit App: https://2025ac05213mlassignmentclassification-jhvpvju4zuscwdcx3etxz6.streamlit.app/
- GitHub Repository: https://github.com/19utpal/2025AC05213_ML_ASSIGNMENT_CLASSIFICATION_.git