from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SDSS Stellar Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_FILE = Path(__file__).with_name("test_csv.csv")
BASE_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
FEATURES = BASE_FEATURES + ["u_g_color", "g_r_color", "r_i_color", "i_z_color"]
CLASS_NAMES = ["GALAXY", "STAR", "QSO"]


st.markdown(
    """
    <style>
        .stApp { background: #f6f3ea; color: #1d2528; }
        h1, h2, h3 { color: #173f46; letter-spacing: 0; }
        div[data-testid="stSidebar"] { background: #e8dfcd; }
        div[data-testid="stMetric"] {
            background: #fffaf0;
            border: 1px solid #d8cab0;
            border-radius: 8px;
            padding: 14px;
        }
        .stButton > button {
            background: #196b69;
            color: white;
            border-radius: 6px;
            border: 0;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    processed = data.copy()

    missing_base_features = [column for column in BASE_FEATURES if column not in processed.columns]
    if missing_base_features:
        missing = ", ".join(missing_base_features)
        raise ValueError(f"Missing required columns: {missing}")

    processed["u_g_color"] = processed["u"] - processed["g"]
    processed["g_r_color"] = processed["g"] - processed["r"]
    processed["r_i_color"] = processed["r"] - processed["i"]
    processed["i_z_color"] = processed["i"] - processed["z"]
    return processed


@st.cache_data
def build_reference_model(reference_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    reference_data = add_engineered_features(reference_data)
    reference_features = reference_data[FEATURES].astype(float)
    means = reference_features.mean()
    standard_deviations = reference_features.std().replace(0, 1)
    scaled_features = (reference_features - means) / standard_deviations
    return scaled_features, means, standard_deviations


def predict_with_knn(
    input_data: pd.DataFrame,
    reference_features: pd.DataFrame,
    reference_labels: pd.Series,
    means: pd.Series,
    standard_deviations: pd.Series,
    neighbor_count: int,
) -> tuple[list[str], pd.DataFrame]:
    input_features = add_engineered_features(input_data)[FEATURES].astype(float)
    scaled_input = ((input_features - means) / standard_deviations).to_numpy()
    scaled_reference = reference_features.to_numpy()
    labels = reference_labels.to_numpy()

    predictions = []
    probabilities = []

    for row in scaled_input:
        distances = np.sqrt(np.sum((scaled_reference - row) ** 2, axis=1))
        neighbor_indexes = np.argsort(distances)[:neighbor_count]
        neighbor_distances = distances[neighbor_indexes]
        neighbor_labels = labels[neighbor_indexes]
        weights = 1 / (neighbor_distances + 1e-9)

        scores = {class_name: 0.0 for class_name in CLASS_NAMES}
        for label, weight in zip(neighbor_labels, weights):
            scores[label] += float(weight)

        total_score = sum(scores.values()) or 1.0
        probabilities.append({class_name: scores[class_name] / total_score for class_name in CLASS_NAMES})
        predictions.append(max(scores, key=scores.get))

    return predictions, pd.DataFrame(probabilities)


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    confusion = pd.DataFrame(0, index=CLASS_NAMES, columns=CLASS_NAMES)
    for true_label, predicted_label in zip(actual, predicted):
        if true_label in CLASS_NAMES and predicted_label in CLASS_NAMES:
            confusion.loc[true_label, predicted_label] += 1

    total = int(confusion.to_numpy().sum())
    correct = int(np.trace(confusion.to_numpy()))
    accuracy = correct / total if total else 0.0

    precision_values = []
    recall_values = []
    f1_values = []

    for class_name in CLASS_NAMES:
        true_positive = confusion.loc[class_name, class_name]
        false_positive = confusion[class_name].sum() - true_positive
        false_negative = confusion.loc[class_name].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        class_weight = confusion.loc[class_name].sum() / total if total else 0.0
        precision_values.append(precision * class_weight)
        recall_values.append(recall * class_weight)
        f1_values.append(f1_score * class_weight)

    return (
        {
            "Accuracy": accuracy,
            "Precision": sum(precision_values),
            "Recall": sum(recall_values),
            "F1 Score": sum(f1_values),
        },
        confusion,
    )


def show_metrics(metrics: dict[str, float]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, f"{value:.4f}")


reference_data = load_reference_data()
reference_features, feature_means, feature_standard_deviations = build_reference_model(reference_data)

st.title("SDSS Stellar Classification")
st.caption("Single-file Streamlit Cloud app using one local CSV as the reference dataset.")

st.sidebar.header("Controls")
neighbor_count = st.sidebar.slider("Nearest neighbors", min_value=3, max_value=25, value=7, step=2)
data_source = st.sidebar.radio("Dataset", ["Use included test_csv.csv", "Upload another CSV"])

st.sidebar.markdown("Required columns: `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`. Add `class` to calculate metrics.")

if data_source == "Upload another CSV":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is None:
        st.info("Upload a CSV to classify astronomical records.")
        st.stop()
    active_data = pd.read_csv(uploaded_file)
else:
    active_data = reference_data.copy()

try:
    predictions, probabilities = predict_with_knn(
        active_data,
        reference_features,
        reference_data["class"],
        feature_means,
        feature_standard_deviations,
        neighbor_count,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

result_data = active_data.copy()
result_data["predicted_class"] = predictions
for class_name in CLASS_NAMES:
    result_data[f"probability_{class_name.lower()}"] = probabilities[class_name]

overview, predictions_tab, data_tab = st.tabs(["Overview", "Predictions", "Data"])

with overview:
    st.subheader("Reference Dataset")
    col_rows, col_features, col_classes = st.columns(3)
    col_rows.metric("Rows", f"{len(reference_data):,}")
    col_features.metric("Features", len(FEATURES))
    col_classes.metric("Classes", len(CLASS_NAMES))

    if "class" in active_data.columns:
        clean_actual = active_data["class"].astype(str).str.upper()
        metrics, confusion = classification_metrics(clean_actual, pd.Series(predictions))
        st.subheader("Evaluation Metrics")
        show_metrics(metrics)
        st.subheader("Confusion Matrix")
        st.dataframe(confusion, use_container_width=True)
    else:
        st.info("No `class` column found, so this run shows predictions only.")

    st.subheader("Predicted Class Mix")
    st.bar_chart(result_data["predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0))

with predictions_tab:
    st.subheader("Classified Records")
    st.dataframe(result_data, use_container_width=True)
    st.download_button(
        "Download predictions",
        result_data.to_csv(index=False),
        file_name="sdss_predictions.csv",
        mime="text/csv",
    )

with data_tab:
    st.subheader("Included CSV Preview")
    st.dataframe(reference_data.head(100), use_container_width=True)
    st.subheader("Feature Summary")
    st.dataframe(add_engineered_features(reference_data)[FEATURES].describe(), use_container_width=True)