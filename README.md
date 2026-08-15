# Sloan Digital Sky Survey (SDSS) Stellar Classification Project

An end-to-end Machine Learning pipeline utilizing the Sloan Digital Sky Survey (SDSS) DR17 dataset to classify cosmic targets into GALAXY, STAR, or QSO (Quasars). Features a robust 12-dimensional engineered model suite and a high-fidelity Streamlit cloud portal interface.

---

## a. Problem Statement
Determining the classification of star systems, galaxies, and super-active galactic nuclei (Quasars) from optical telescope parameters is a foundational task in astrophysics. The challenge is to reconstruct precise class maps from astronomical positions, optical magnitudes, and cosmic redshifts. Real-time categorization of high-volume telescope feeds requires automated classifiers that outperform traditional hand-tuned astronomical filters on correctness, speed, and cross-metric validation.

---

## b. Dataset Description
The model is trained on the public **Sloan Digital Sky Survey (SDSS) DR17** cosmic observations catalog.
- **Instance Count**: 100,000 observations (min instance size 500 requirement met).
- **Features Size**: Exactly 12 active physical characteristics (min feature size 12 requirement met).
- **Target Classes**: `GALAXY` (0), `STAR` (1), `QSO` (Quasar) (2).

### Variables Dictionary:
1. `alpha`: Right Ascension angle (Celestial Position).
2. `delta`: Declination angle (Celestial Position).
3. `u`: Ultraviolet Filter Band magnitude.
4. `g`: Green Filter Band magnitude.
5. `r`: Red Filter Band magnitude.
6. `i`: Near Infrared Filter Band magnitude.
7. `z`: Far Infrared Filter Band magnitude.
8. `redshift`: Wavelength expansion shift index based on relative velocity (strongly correlates to distance).
9. `u_g_color` (*Engineered*): Difference between Ultraviolet (u) and Green (g) bands.
10. `g_r_color` (*Engineered*): Difference between Green (g) and Red (r) bands.
11. `r_i_color` (*Engineered*): Difference between Red (r) and Near Infrared (i) bands.
12. `i_z_color` (*Engineered*): Difference between Near Infrared (i) and Far Infrared (z) bands.

---

## c. GitHub Repository Link
The complete workspace is hosted at:
[GitHub Repository - Astronomical Star Classification](https://github.com/your-username/star_classification_project)

---

## d. Models Used and Evaluation Comparisons

The comparison metrics below are calculated on the 20% stratified evaluation partition (20,000 samples) during the model runtime pipeline.

### Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9565 | 0.9865 | 0.9565 | 0.9565 | 0.9561 | 0.9229 |
| **Decision Tree** | 0.9771 | 0.9858 | 0.9769 | 0.9771 | 0.9769 | 0.9592 |
| **kNN** | 0.9495 | 0.9786 | 0.9496 | 0.9495 | 0.9494 | 0.9102 |
| **Naive Bayes** | 0.7255 | 0.9171 | 0.7791 | 0.7255 | 0.6467 | 0.5097 |
| **Random Forest (Ensemble)** | 0.9801 | 0.9964 | 0.9800 | 0.9801 | 0.9800 | 0.9647 |
| **Gradient Boosting (Ensemble)** | 0.9806 | 0.9969 | 0.9805 | 0.9806 | 0.9805 | 0.9654 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly with **95.65% Accuracy** once magnitudes and celestial positions were normalized using StandardScaler. Represents a robust linear boundary baseline, but suffers slightly when handling complex non-linear combinations of magnitude curves. |
| **Decision Tree** | Achieved excellent results (**97.71% Accuracy**). Decouples feature scales naturally. Pruning the maximum depth at 10 protected the tree from splitting on regional noise while remaining highly responsive to structural parameters. |
| **kNN** | Delivered robust results (**94.95% Accuracy**). While sensitive to the scaled coordinates, performance remains very solid thanks to the custom color band engineered index combinations. |
| **Naive Bayes** | Recorded the lowest performance (**72.55% Accuracy**, **0.5097 MCC**). This is primarily driven by the "feature independence" assumption of Naive Bayes, which is violated by the highly correlated photometric filters ($u, g, r, i, z$) and our engineered color indexes. |
| **Random Forest** | Outstanding classification performance (**98.01% Accuracy**). Constructing an ensemble of 100 estimators naturally smoothed high individual variance and delivered reliable probability boundaries. |
| **Gradient Boosting** | Achieved peak accuracy at **98.06%** with a **0.9654 MCC Score**. Sequential learning on error residuals proved heavily responsive to subtle features, particularly in distinguishing distant Quasars (QSOs) from bright local stars. |

### Overall Winner for your dataset?
**Gradient Boosting (HistGradientBoostingClassifier)** is the overall winner. It achieved the highest overall scores across every benchmark metric, including a peak Accuracy of **98.06%** and an MCC Score of **0.9654**, indicating extremely stable class predictions.

---

## How to Run locally on BITS Virtual Lab

1. Place `star_classification.csv` in your environment path or specify its target path.
2. Execute the model execution and training pipeline python file using:
   ```bash
   python train_models.py
   ```
3. Boot up the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
