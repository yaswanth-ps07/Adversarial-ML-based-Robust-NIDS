import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import *
from src.model import load_model
from src.evaluate import predict, attack_array
from src.metrics import calculate_metrics


st.set_page_config(
    page_title="Robust NIDS",
    page_icon="🛡️",
    layout="wide"
)

st.title(
    "Improving the Robustness of Deep Learning-Based "
    "Network Intrusion Detection Against Adversarial Attacks"
)

page = st.sidebar.radio(
    "Section",
    [
        "Home",
        "Dataset",
        "Baseline Model",
        "Adversarial Attack",
        "Robust Model",
        "Comparison",
        "SHAP Explainability",
        "Prediction"
    ]
)


def artifacts_ready():
    return (
        (ARTIFACT_DIR / "splits.npz").exists()
        and (MODEL_DIR / "baseline_dnn.pt").exists()
    )


def metric_card(m):
    cols = st.columns(4)

    for c, (k, v) in zip(
        cols,
        [
            ("Accuracy", m["accuracy"]),
            ("Precision", m["precision"]),
            ("Recall", m["recall"]),
            ("F1", m["f1"])
        ]
    ):
        c.metric(k, f"{v:.2%}")


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.subheader("Project Overview")

    st.write(
        "This project evaluates whether a deep neural network-based "
        "network intrusion detection system can remain effective "
        "when its input features are deliberately perturbed by "
        "adversarial attacks."
    )

    st.code(
        "CIC-IDS2017 → preprocessing → DNN → clean evaluation → "
        "FGSM/PGD → adversarial training → robust DNN → comparison"
    )

    st.info(
        "FGSM and PGD are controlled feature-space attacks used for "
        "model robustness evaluation. They do not generate or transmit "
        "malicious network packets."
    )

    # --------------------------------------------------------
    # Results overview
    # --------------------------------------------------------

    comparison_file = METRIC_DIR / "comparison.csv"

    if comparison_file.exists():

        df = pd.read_csv(comparison_file)

        st.subheader("Results Overview")

        baseline_clean = df[df["scenario"] == "Baseline Clean"]
        robust_clean = df[df["scenario"] == "Robust Clean"]
        baseline_fgsm = df[df["scenario"] == "Baseline FGSM"]
        robust_fgsm = df[df["scenario"] == "Robust FGSM"]
        baseline_pgd = df[df["scenario"] == "Baseline PGD"]
        robust_pgd = df[df["scenario"] == "Robust PGD"]

        if not baseline_clean.empty and not robust_clean.empty:

            st.markdown("### Clean Traffic Performance")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Baseline Accuracy",
                f"{baseline_clean.iloc[0]['accuracy']:.2%}"
            )

            c2.metric(
                "Robust Accuracy",
                f"{robust_clean.iloc[0]['accuracy']:.2%}"
            )

            c3.metric(
                "Accuracy Change",
                f"{(
                    robust_clean.iloc[0]['accuracy']
                    - baseline_clean.iloc[0]['accuracy']
                ):+.2%}"
            )

        st.markdown("### Adversarial Robustness")

        results = []

        scenarios = [
            ("FGSM", baseline_fgsm, robust_fgsm),
            ("PGD", baseline_pgd, robust_pgd)
        ]

        for attack_name, baseline, robust in scenarios:

            if not baseline.empty and not robust.empty:

                results.append(
                    {
                        "Attack": attack_name,
                        "Baseline Accuracy": baseline.iloc[0]["accuracy"],
                        "Robust Accuracy": robust.iloc[0]["accuracy"],
                        "Baseline F1": baseline.iloc[0]["f1"],
                        "Robust F1": robust.iloc[0]["f1"],
                        "Baseline ASR": baseline.iloc[0]["attack_success_rate"],
                        "Robust ASR": robust.iloc[0]["attack_success_rate"]
                    }
                )

        if results:

            result_df = pd.DataFrame(results)

            display_df = result_df.copy()

            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].map(
                    lambda x: f"{x:.2%}"
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Key Finding")

            if not robust_fgsm.empty and not baseline_fgsm.empty:

                fgsm_gain = (
                    robust_fgsm.iloc[0]["accuracy"]
                    - baseline_fgsm.iloc[0]["accuracy"]
                )

                st.success(
                    f"Adversarial training improved FGSM accuracy by "
                    f"{fgsm_gain:.2%}, while maintaining approximately "
                    f"the same clean-data accuracy."
                )

            if not robust_pgd.empty:

                pgd_accuracy = robust_pgd.iloc[0]["accuracy"]
                pgd_asr = robust_pgd.iloc[0]["attack_success_rate"]

                st.warning(
                    f"PGD remains the stronger challenge: the robust model "
                    f"achieves {pgd_accuracy:.2%} accuracy with "
                    f"{pgd_asr:.2%} attack success rate."
                )

        st.markdown("### Accuracy Comparison")

        chart_data = df[
            df["scenario"].isin(
                [
                    "Baseline Clean",
                    "Baseline FGSM",
                    "Baseline PGD",
                    "Robust Clean",
                    "Robust FGSM",
                    "Robust PGD"
                ]
            )
        ][["scenario", "accuracy"]].set_index("scenario")

        st.bar_chart(chart_data)

    else:

        st.warning(
            "Evaluation results are not available yet. "
            "Run the training and evaluation pipeline first."
        )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    p = ARTIFACT_DIR / "dataset_metadata.json"

    if p.exists():

        d = json.loads(p.read_text())

        c1, c2 = st.columns(2)

        c1.metric("Loaded Samples", f"{d['rows_loaded']:,}")
        c2.metric("Features", d["features"])

        st.subheader("Class Distribution")

        st.bar_chart(
            pd.DataFrame(
                {
                    "Class": ["Normal", "Attack"],
                    "Samples": [d["normal"], d["attack"]]
                }
            ).set_index("Class")
        )

    else:

        st.warning(
            "Run python src/train.py after placing CIC-IDS2017 "
            "CSV files in data/raw/."
        )


# ============================================================
# BASELINE / ROBUST MODEL
# ============================================================

elif page in ["Baseline Model", "Robust Model"]:

    p = METRIC_DIR / "comparison.csv"

    if not p.exists():

        st.warning(
            "Run training/evaluation commands first."
        )

    else:

        df = pd.read_csv(p)

        key = (
            "Baseline Clean"
            if page == "Baseline Model"
            else "Robust Clean"
        )

        row = df[df.scenario == key]

        if row.empty:

            st.warning(
                "This model has not been evaluated yet."
            )

        else:

            st.subheader(f"{page} — Clean Performance")

            metric_card(row.iloc[0].to_dict())

            img = FIGURE_DIR / (
                f'{key.lower().replace(" ", "_")}_cm.png'
            )

            if img.exists():
                st.image(
                    str(img),
                    caption=f"{key} Confusion Matrix"
                )


# ============================================================
# ADVERSARIAL ATTACK
# ============================================================

elif page == "Adversarial Attack":

    if not artifacts_ready():

        st.warning(
            "Train the baseline model first."
        )

    else:

        st.subheader("Interactive Adversarial Attack")

        kind = st.selectbox(
            "Attack",
            ["FGSM", "PGD"]
        )

        eps = st.slider(
            "Epsilon",
            0.01,
            0.30,
            0.10,
            0.01
        )

        count = st.slider(
            "Samples",
            100,
            5000,
            1000,
            100
        )

        if st.button("Generate and Evaluate"):

            z = np.load(
                ARTIFACT_DIR / "splits.npz"
            )

            X = z["X_test"][:count]
            y = z["y_test"][:count]

            model = load_model(
                MODEL_DIR / "baseline_dnn.pt",
                X.shape[1]
            )

            clean = predict(model, X)

            adv = attack_array(
                model,
                X,
                y,
                kind.lower(),
                eps
            )

            pred = predict(model, adv)

            m, _, _ = calculate_metrics(
                y,
                pred,
                clean
            )

            metric_card(m)

            st.metric(
                "Attack Success Rate",
                f"{m['attack_success_rate']:.2%}"
            )


# ============================================================
# COMPARISON
# ============================================================

elif page == "Comparison":

    p = METRIC_DIR / "comparison.csv"

    if not p.exists():

        st.warning(
            "Run training/evaluation commands first."
        )

    else:

        df = pd.read_csv(p)

        st.subheader("Baseline vs Robust Model")

        # ----------------------------------------------------
        # Key performance metrics
        # ----------------------------------------------------

        st.markdown("### Performance Summary")

        summary_rows = []

        for scenario in [
            "Baseline Clean",
            "Baseline FGSM",
            "Baseline PGD",
            "Robust Clean",
            "Robust FGSM",
            "Robust PGD"
        ]:

            row = df[df["scenario"] == scenario]

            if not row.empty:

                r = row.iloc[0]

                summary_rows.append(
                    {
                        "Scenario": scenario,
                        "Accuracy": r["accuracy"],
                        "Precision": r["precision"],
                        "Recall": r["recall"],
                        "F1 Score": r["f1"],
                        "False Positive Rate": r["false_positive_rate"],
                        "Attack Success Rate":
                            r["attack_success_rate"]
                    }
                )

        summary_df = pd.DataFrame(summary_rows)

        display_df = summary_df.copy()

        percentage_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Positive Rate",
            "Attack Success Rate"
        ]

        for col in percentage_columns:

            display_df[col] = display_df[col].map(
                lambda x:
                "-" if pd.isna(x) else f"{x:.2%}"
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        st.markdown("### Accuracy")

        accuracy_df = summary_df[
            ["Scenario", "Accuracy"]
        ].set_index("Scenario")

        st.bar_chart(accuracy_df)

        # ----------------------------------------------------
        # F1 Score
        # ----------------------------------------------------

        st.markdown("### F1 Score")

        f1_df = summary_df[
            ["Scenario", "F1 Score"]
        ].set_index("Scenario")

        st.bar_chart(f1_df)

        # ----------------------------------------------------
        # Attack Success Rate
        # ----------------------------------------------------

        st.markdown("### Attack Success Rate")

        asr_df = summary_df.dropna(
            subset=["Attack Success Rate"]
        )

        asr_df = asr_df[
            ["Scenario", "Attack Success Rate"]
        ].set_index("Scenario")

        st.bar_chart(asr_df)

        # ----------------------------------------------------
        # Main conclusion
        # ----------------------------------------------------

        st.markdown("### Robustness Conclusion")

        baseline_fgsm = df[
            df["scenario"] == "Baseline FGSM"
        ]

        robust_fgsm = df[
            df["scenario"] == "Robust FGSM"
        ]

        baseline_pgd = df[
            df["scenario"] == "Baseline PGD"
        ]

        robust_pgd = df[
            df["scenario"] == "Robust PGD"
        ]

        if (
            not baseline_fgsm.empty
            and not robust_fgsm.empty
            and not baseline_pgd.empty
            and not robust_pgd.empty
        ):

            fgsm_gain = (
                robust_fgsm.iloc[0]["accuracy"]
                - baseline_fgsm.iloc[0]["accuracy"]
            )

            pgd_gain = (
                robust_pgd.iloc[0]["accuracy"]
                - baseline_pgd.iloc[0]["accuracy"]
            )

            fgsm_asr_reduction = (
                baseline_fgsm.iloc[0]["attack_success_rate"]
                - robust_fgsm.iloc[0]["attack_success_rate"]
            )

            pgd_asr_reduction = (
                baseline_pgd.iloc[0]["attack_success_rate"]
                - robust_pgd.iloc[0]["attack_success_rate"]
            )

            st.success(
                f"FGSM accuracy improved by {fgsm_gain:.2%}, "
                f"while attack success decreased by "
                f"{fgsm_asr_reduction:.2%}."
            )

            st.success(
                f"PGD accuracy improved by {pgd_gain:.2%}, "
                f"while attack success decreased by "
                f"{pgd_asr_reduction:.2%}."
            )

            st.info(
                "The robust model maintains similar clean-data "
                "performance while providing substantially stronger "
                "resistance to adversarial perturbations. PGD remains "
                "the more challenging attack."
            )

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

elif page == "SHAP Explainability":

    st.subheader("SHAP Feature Importance")

    shap_file = METRIC_DIR / "shap_feature_importance.csv"

    if not shap_file.exists():

        st.warning(
            "SHAP results are not available. "
            "Run: python .\\src\\explain.py"
        )

    else:

        shap_df = pd.read_csv(shap_file)

        st.write(
            "SHAP (SHapley Additive exPlanations) estimates how "
            "strongly each network-flow feature contributes to "
            "the model's predictions. Higher mean absolute SHAP "
            "values indicate greater overall feature influence."
        )

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(20, len(shap_df)),
            value=min(10, len(shap_df)),
            step=1
        )

        top_features = shap_df.head(top_n).copy()

        st.markdown("### Top Features")

        display_shap = top_features.copy()

        display_shap["mean_abs_shap"] = (
            display_shap["mean_abs_shap"].map(
                lambda x: f"{x:.4f}"
            )
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Feature Importance")

        chart_df = (
            top_features
            .set_index("feature")[
                ["mean_abs_shap"]
            ]
            .sort_values("mean_abs_shap")
        )

        st.bar_chart(chart_df)

        st.info(
            "These rankings describe feature importance for the "
            "trained baseline DNN on the evaluated test samples. "
            "They do not by themselves indicate whether a feature "
            "increases or decreases the probability of an attack."
        )

# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    if not artifacts_ready():

        st.warning(
            "Train the baseline model first."
        )

    else:

        st.subheader("Network Intrusion Prediction")

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        model_options = ["Baseline DNN"]

        if (MODEL_DIR / "robust_dnn.pt").exists():
            model_options.append("Robust DNN")

        selected_model = st.selectbox(
            "Select Model",
            model_options
        )

        if selected_model == "Robust DNN":
            model_path = MODEL_DIR / "robust_dnn.pt"
        else:
            model_path = MODEL_DIR / "baseline_dnn.pt"

        # ----------------------------------------------------
        # Load preprocessing artifacts
        # ----------------------------------------------------

        features = json.loads(
            (
                ARTIFACT_DIR / "feature_names.json"
            ).read_text()
        )

        scaler = joblib.load(
            ARTIFACT_DIR
            / "scaler"
            / "standard_scaler.joblib"
        )

        medians = joblib.load(
            ARTIFACT_DIR
            / "scaler"
            / "training_medians.joblib"
        )

        z = np.load(
            ARTIFACT_DIR / "splits.npz"
        )

        # ----------------------------------------------------
        # Test-set sample prediction
        # ----------------------------------------------------

        st.markdown("### Test Sample")

        sample_id = st.number_input(
            "Select test sample index",
            min_value=0,
            max_value=len(z["X_test"]) - 1,
            value=0,
            step=1
        )

        if st.button("Predict Selected Sample"):

            model = load_model(
                model_path,
                len(features)
            )

            prediction = predict(
                model,
                z["X_test"][[sample_id]]
            )[0]

            if prediction:
                st.error(
                    f"{selected_model} Prediction: ATTACK"
                )
            else:
                st.success(
                    f"{selected_model} Prediction: NORMAL"
                )

        # ----------------------------------------------------
        # CSV upload prediction
        # ----------------------------------------------------

        st.markdown("### Upload a Network Flow")

        st.write(
            "Upload a one-row CSV containing the same "
            "network-flow feature columns used during training."
        )

        uploaded = st.file_uploader(
            "Choose CSV file",
            type="csv"
        )

        if uploaded:

            raw = pd.read_csv(uploaded)

            missing = [
                f for f in features
                if f not in raw.columns
            ]

            if missing:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing[:10])
                )

            else:

                row = (
                    raw[features]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace(
                        [np.inf, -np.inf],
                        np.nan
                    )
                    .fillna(medians)
                )

                x = scaler.transform(
                    row
                ).astype("float32")

                model = load_model(
                    model_path,
                    len(features)
                )

                prediction = predict(
                    model,
                    x
                )[0]

                if prediction:

                    st.error(
                        f"{selected_model} Prediction: ATTACK"
                    )

                else:

                    st.success(
                        f"{selected_model} Prediction: NORMAL"
                    )