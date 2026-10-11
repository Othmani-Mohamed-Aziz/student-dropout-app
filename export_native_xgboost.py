"""Export Colab-trained models in a Windows-safe format.

Run this in Colab after ``tuned_models`` has been created or loaded. The
non-XGBoost pipelines remain joblib files; XGBoost estimators are saved using
their native JSON format and reconstructed by app.py.
"""

from pathlib import Path
import json
import shutil

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


def export_native_artifacts(models, output_dir="artifacts"):
    output = Path(output_dir)
    xgb_dir = output / "xgboost_native"
    output.mkdir(parents=True, exist_ok=True)
    if xgb_dir.exists():
        shutil.rmtree(xgb_dir)
    xgb_dir.mkdir()

    portable_models = {}
    manifest = {
        "format": "native-xgboost-v1",
        "portable_models": "tuned_models_non_xgb.joblib",
        "xgboost_models": [],
    }

    for key, fitted in models.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Expected model key (scenario, model_name), got {key!r}")
        scenario, model_name = key
        if model_name != "XGBoost":
            portable_models[key] = fitted
            continue

        if not isinstance(fitted, Pipeline) or not fitted.steps:
            raise TypeError(f"XGBoost entry {key!r} must be a fitted sklearn Pipeline")
        model_step_name, estimator = fitted.steps[-1]
        if not hasattr(estimator, "get_booster") or not hasattr(estimator, "save_model"):
            raise TypeError(f"Final estimator for {key!r} is not an XGBoost estimator")

        index = len(manifest["xgboost_models"])
        native_name = f"model_{index}.json"
        preprocessor_name = f"preprocessor_{index}.joblib"
        estimator.save_model(str(xgb_dir / native_name))

        if len(fitted.steps) > 1:
            joblib.dump(Pipeline(fitted.steps[:-1]), xgb_dir / preprocessor_name, compress=3)
        else:
            preprocessor_name = None

        classes = getattr(estimator, "classes_", np.array([0, 1]))
        manifest["xgboost_models"].append({
            "scenario": scenario,
            "model_name": model_name,
            "model_file": f"xgboost_native/{native_name}",
            "preprocessor_file": (
                f"xgboost_native/{preprocessor_name}" if preprocessor_name else None
            ),
            "model_step_name": model_step_name,
            "classes": np.asarray(classes).tolist(),
        })

    joblib.dump(portable_models, output / manifest["portable_models"], compress=3)
    (output / "native_models_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    raise SystemExit("Import export_native_artifacts in Colab and call it with tuned_models.")
