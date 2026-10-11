from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
SCENARIOS = ["Scenario A - Early", "Scenario B - After 1st semester"]
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]

st.set_page_config(page_title="Student Dropout Risk", page_icon="🎓", layout="wide")

@st.cache_resource
def load_artifacts(artifact_signature=()):
    mp, jp = ARTIFACTS / "tuned_models.joblib", ARTIFACTS / "metadata.json"
    try:
        with jp.open(encoding="utf-8") as f:
            metadata = json.load(f)

        manifest_path = ARTIFACTS / "native_models_manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            models = joblib.load(ARTIFACTS / manifest["portable_models"])
            from xgboost import XGBClassifier

            for item in manifest["xgboost_models"]:
                model = XGBClassifier()
                model.load_model(str(ARTIFACTS / item["model_file"]))
                if not hasattr(model, "n_classes_"):
                    model.n_classes_ = len(item.get("classes", [0, 1]))
                preprocessor_file = item.get("preprocessor_file")
                if preprocessor_file:
                    preprocessor = joblib.load(ARTIFACTS / preprocessor_file)
                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        (item.get("model_step_name", "model"), model),
                    ])
                models[(item["scenario"], item["model_name"])] = model
            return models, metadata, None

        if not mp.exists() or not jp.exists():
            return None, None, "Missing tuned_models.joblib or metadata.json in artifacts/"
        return joblib.load(mp), metadata, None
    except Exception as e:
        return None, None, str(e)

def artifact_signature():
    if not ARTIFACTS.exists():
        return ()
    return tuple(
        (str(path.relative_to(ARTIFACTS)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(ARTIFACTS.rglob("*"))
        if path.is_file()
    )


model_path = ARTIFACTS / "tuned_models.joblib"
metadata_path = ARTIFACTS / "metadata.json"
models, metadata, artifact_error = load_artifacts(
    artifact_signature(),
)

def scenario_metadata(scenario):
    return metadata.get("scenarios", {}).get(scenario, {})

def get_threshold(scenario, model_name):
    if model_name == "XGBoost":
        return float(metadata.get("xgboost_thresholds", {}).get(scenario, 0.5))
    return 0.5

def predict(frame, scenario, model_name, threshold):
    features = scenario_metadata(scenario).get("features", [])
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = frame.loc[:, features].copy()
    # The saved fitted pipeline performs preprocessing itself.
    probabilities = models[(scenario, model_name)].predict_proba(X)[:, 1]
    classes = np.where(probabilities >= threshold, "Dropout", "Non-Dropout")
    return pd.DataFrame({
        "dropout_probability": probabilities,
        "predicted_class": classes,
        "threshold_used": threshold
    }, index=frame.index)

def overview():
    st.markdown("""
    # 🎓 Student Dropout Risk
    Interactive machine-learning workbench for the UCI student dropout classification project.
    """)
    a, b, c = st.columns(3)
    a.metric("Prediction scenarios", "2")
    b.metric("Model families", "5")
    c.metric("Artifacts", "Loaded" if models else "Pending")
    st.info("Positive class = Dropout. Non-Dropout combines Graduate and Enrolled; it does not mean the student will graduate.")
    st.subheader("Scenarios")
    st.markdown("""
    - **Scenario A — Early:** admission, demographic and administrative features, without first-semester results.
    - **Scenario B — After 1st semester:** adds first-semester academic indicators.

    Ensure every input feature is available at the intended prediction time.
    """)

def data_explorer():
    st.title("Data Explorer")
    uploaded = st.file_uploader("Upload dataset CSV", type="csv", key="eda")
    if not uploaded:
        st.caption("Upload the dataset to inspect rows, data quality, target balance and feature distributions.")
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    c.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)
    quality = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique()
    })
    st.subheader("Data quality")
    st.dataframe(quality, use_container_width=True)
    if "Target" in df:
        st.subheader("Target distribution")
        st.bar_chart(df["Target"].value_counts())
    feature = st.selectbox("Explore a feature", df.columns)
    if pd.api.types.is_numeric_dtype(df[feature]):
        st.bar_chart(df[feature].value_counts().sort_index().head(100))
        st.write(df[feature].describe())
    else:
        st.bar_chart(df[feature].astype(str).value_counts().head(30))

def model_evaluation():
    st.title("Model Evaluation")
    candidates = [
        ARTIFACTS / "tuned_models_test_results.csv",
        ARTIFACTS / "baseline_model_comparison.csv",
        ARTIFACTS / "xgboost_threshold_selected_test.csv"
    ]
    paths = [p for p in candidates if p.exists()]
    if not paths:
        st.info("Add exported evaluation CSV files to the artifacts/ folder.")
        return
    chosen = st.selectbox("Results file", paths, format_func=lambda p: p.name)
    df = pd.read_csv(chosen)
    st.dataframe(df, use_container_width=True, hide_index=True)
    model_col = next((c for c in df if c.lower() in ("model", "model_name")), None)
    scenario_col = next((c for c in df if c.lower() == "scenario"), None)
    auc_col = next((c for c in df if c.lower().replace("-", "_") in ("roc_auc",)), None)
    if model_col and auc_col:
        if scenario_col:
            scenario = st.selectbox("Scenario", df[scenario_col].dropna().unique())
            plot = df[df[scenario_col] == scenario]
        else:
            plot = df
        st.subheader("ROC-AUC")
        st.bar_chart(plot[[model_col, auc_col]].dropna().set_index(model_col))

def prediction_page():
    st.title("Student Prediction")
    if models is None or metadata is None:
        model_path = ARTIFACTS / "tuned_models.joblib"
        metadata_path = ARTIFACTS / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            st.warning("Model artifacts are not installed. Export tuned_models.joblib and metadata.json from Colab into the artifacts/ folder.")
            st.code("artifacts/tuned_models.joblib\nartifacts/metadata.json")
        else:
            st.error("Model artifacts were found, but tuned_models.joblib could not be loaded.")
            st.code(artifact_error or "Unknown artifact loading error")
            st.info("Use the same Python/scikit-learn versions used when the model was exported, then restart Streamlit.")
        return
    scenario = st.selectbox("Prediction scenario", SCENARIOS)
    available = [m for m in MODEL_ORDER if (scenario, m) in models]
    if not available:
        st.error(f"No models found for {scenario}. Found keys: {list(models.keys())}")
        return
    default_index = available.index("XGBoost") if "XGBoost" in available else 0
    model_name = st.selectbox("Model", available, index=default_index)
    default_threshold = get_threshold(scenario, model_name)
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(np.clip(default_threshold, .05, .95)), .01)
    meta = scenario_metadata(scenario)
    features = meta.get("features", [])
    numeric = set(meta.get("numeric_features", []))
    categorical = set(meta.get("categorical_features", []))
    if not features:
        st.error("No feature metadata found in metadata.json.")
        return

    single, batch = st.tabs(["Manual entry", "CSV batch upload"])
    with single:
        st.write("Categorical fields use the original numeric codes from the dataset.")
        with st.form("single_student"):
            values = {}
            columns = st.columns(2)
            for i, feature in enumerate(features):
                with columns[i % 2]:
                    if feature in categorical:
                        values[feature] = st.number_input(feature, value=0, step=1, format="%d")
                    else:
                        values[feature] = st.number_input(feature, value=0.0, format="%.4f")
            submit = st.form_submit_button("Predict dropout risk", type="primary", use_container_width=True)
        if submit:
            try:
                result = predict(pd.DataFrame([values]), scenario, model_name, threshold)
                p = float(result["dropout_probability"].iloc[0])
                label = result["predicted_class"].iloc[0]
                a, b = st.columns(2)
                a.metric("Predicted class", label)
                b.metric("Estimated dropout probability", f"{p:.1%}")
                st.progress(float(np.clip(p, 0, 1)))
                if label == "Dropout":
                    st.warning("Potential dropout risk flagged. Treat this as a screening signal, not a definitive judgment.")
                else:
                    st.success("Classified as non-dropout at this threshold; this does not guarantee graduation.")
                st.download_button("Download prediction CSV", result.to_csv(index=False).encode(), "student_prediction.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with batch:
        st.write("Upload a CSV with one student per row and all required feature columns.")
        st.caption("Column names and category codes must match the training dataset. Blank values are passed to the fitted pipeline's imputers.")
        file = st.file_uploader("Students CSV", type="csv", key="batch")
        if file:
            try:
                df = pd.read_csv(file)
                st.write(f"Rows: {len(df):,}")
                st.dataframe(df.head(10), use_container_width=True)
                missing = [c for c in features if c not in df.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                elif st.button("Run batch predictions", type="primary"):
                    result = predict(df, scenario, model_name, threshold)
                    output = pd.concat([df.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
                    st.success(f"Completed {len(output):,} predictions.")
                    st.dataframe(output.head(100), use_container_width=True)
                    st.download_button("Download all predictions", output.to_csv(index=False).encode(), "student_dropout_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

def about():
    st.title("About")
    st.markdown("""
    **Models:** Logistic Regression, Decision Tree, KNN, Random Forest and XGBoost.

    **Output:** Dropout versus Non-Dropout (Graduate or Enrolled).

    Predictions are estimates, not causal explanations or certainties. Use them to support human review and student assistance, not as the sole basis for punitive decisions. SHAP/LIME explanations are postponed and are not included in this version.
    """)

with st.sidebar:
    st.title("🎓 Student Risk")
    page = st.radio("Navigation", ["Overview", "Data Explorer", "Model Evaluation", "Student Prediction", "About"])
    st.divider()
    st.caption("Artifacts ready" if models else "Artifacts not installed")

if page == "Overview":
    overview()
elif page == "Data Explorer":
    data_explorer()
elif page == "Model Evaluation":
    model_evaluation()
elif page == "Student Prediction":
    prediction_page()
else:
    about()
