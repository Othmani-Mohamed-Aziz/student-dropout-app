# Student Dropout Risk — Streamlit App

## Features
- Dataset CSV explorer, quality checks, target distribution and feature summaries.
- Evaluation dashboard using saved Colab metric CSVs.
- Early and after-first-semester prediction scenarios.
- Five tuned models, manual single-student form, batch CSV prediction.
- Threshold slider and downloadable prediction results.
- Uses saved fitted pipelines, preserving preprocessing. The app does not train models.
- SHAP/LIME are intentionally postponed.

## Add Colab artifacts
Copy these files from the Colab export into `artifacts/`:
- `tuned_models.joblib`
- `metadata.json`
- Optional: `tuned_models_test_results.csv`, `baseline_model_comparison.csv`, `xgboost_threshold_selected_test.csv`

### Cross-platform XGBoost export

Colab/Linux XGBoost pipelines should not be unpickled directly on Windows.
Upload `export_native_xgboost.py` to Colab, then run:

```python
from pathlib import Path
import shutil
from google.colab import files
from export_native_xgboost import export_native_artifacts

out = Path("/content/native_export")
if out.exists():
    shutil.rmtree(out)
export_native_artifacts(tuned_models, out)
shutil.copy("/content/artifacts/metadata.json", out / "metadata.json")
shutil.make_archive("native_export", "zip", out)
files.download("/content/native_export.zip")
```

Extract the archive into the app's `artifacts/` folder. The app detects
`native_models_manifest.json`, loads non-XGBoost pipelines from
`tuned_models_non_xgb.joblib`, and loads XGBoost models from native JSON files.

Expected model keys include `("Scenario A - Early", "XGBoost")` and `("Scenario B - After 1st semester", "Logistic Regression")`.

## Run locally
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input requirements
- Manual categorical inputs use original numeric dataset codes.
- Batch CSV must include every required feature column with exact original column names.
- Empty values can be imputed only if the saved pipeline includes appropriate imputers.
- Do not pass `Target` as a feature.

## Notes
The prediction is Dropout versus Non-Dropout (Graduate or Enrolled), not graduation versus non-graduation. Predictions are screening estimates, not certainties. Verify feature availability at the intended prediction time. The current version runs inference in Streamlit; FastAPI can be added as a separate service later. Only load trusted joblib files.
