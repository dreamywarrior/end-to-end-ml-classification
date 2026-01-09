import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Classification Model Evaluator",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("resources/sk_background.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL METADATA
# --------------------------------------------------
MODEL_INFO = {
    "Logistic Regression": (
        "Logistic_Regression.pkl",
        "Linear probabilistic classifier using a sigmoid decision boundary."
    ),
    "Decision Tree": (
        "Decision_Tree.pkl",
        "Tree-based model using feature splits; highly interpretable."
    ),
    "KNN": (
        "KNN.pkl",
        "Distance-based classifier using nearest neighbors."
    ),
    "Naive Bayes": (
        "Naive_Bayes.pkl",
        "Probabilistic classifier assuming conditional independence."
    ),
    "Random Forest": (
        "Random_Forest.pkl",
        "Ensemble of decision trees reducing overfitting."
    ),
    "XGBoost": (
        "XGBoost.pkl",
        "Gradient-boosted trees optimized for performance."
    )
}

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 ML Classification Model Evaluator")

# --------------------------------------------------
# SIDEBAR – DATA UPLOAD
# --------------------------------------------------
st.sidebar.header("📂 Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --------------------------------------------------
# LANDING / INTRO SECTION
# --------------------------------------------------
if uploaded_file is None:
    st.markdown("## 👋 Welcome")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
        ### 🔍 What this app does
        Evaluate and compare **pre-trained classification models**
        on unseen test data.

        **Key Features**
        - Multi-model evaluation
        - Class-wise performance analysis
        - ROC curves & confusion matrices
        - Model comparison dashboard
        - Downloadable reports
        """)

        st.markdown("""
        ### 🧭 How it works
        1. Upload test dataset  
        2. Select models  
        3. Analyze metrics & visuals  
        4. Compare models  
        5. Download results  
        """)

    with col2:
        st.image(
            "resources/model_evaluator.jpg",
            caption="Evaluation Workflow",
            width="stretch"
        )

    st.info("⬅️ Upload test data from the sidebar to begin")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = pd.read_csv(uploaded_file)

with st.expander("🔍 Preview Uploaded Data"):
    st.write("Shape:", data.shape)
    st.dataframe(data.head())

X_test = data.iloc[:, :-1]
y_test = data.iloc[:, -1]

mapping_df = pd.read_csv("model/target_class_encoding.csv")
target_mapping = dict(zip(mapping_df["class"], mapping_df["encoded"]))
inverse_mapping = {v: k for k, v in target_mapping.items()}
y_test_enc = y_test.map(target_mapping)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.sidebar.header("🤖 Model Selection")

selected_models = st.sidebar.multiselect(
    "Select one or more models",
    list(MODEL_INFO.keys()),
    default=["Logistic Regression"]
)

if not selected_models:
    st.warning("Select at least one model")
    st.stop()

# --------------------------------------------------
# STORAGE
# --------------------------------------------------
comparison_results = []

# --------------------------------------------------
# MODEL EVALUATION LOOP
# --------------------------------------------------
for model_name in selected_models:
    st.subheader(f"🤖 {model_name}")
    st.info(MODEL_INFO[model_name][1])

    with open(f"model/{MODEL_INFO[model_name][0]}", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr")
    except:
        y_prob, auc = None, None

    acc = accuracy_score(y_test_enc, y_pred)
    prec = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test_enc, y_pred)

    comparison_results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc
    })

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c1.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c2.metric("F1 Score", f"{f1:.4f}")
    c3.metric("MCC", f"{mcc:.4f}")
    c3.metric("AUC", f"{auc:.4f}" if auc else "N/A")

    # --------------------------------------------------
    # CLASS-WISE METRICS (GRADIENT)
    # --------------------------------------------------
    st.markdown("### 📋 Class-wise Metrics")

    report = classification_report(
        y_test_enc, y_pred,
        target_names=target_mapping.keys(),
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(index="accuracy", errors="ignore")
    report_df = report_df[["precision", "recall", "f1-score"]]

    styled_report = (
        report_df
        .style
        .background_gradient(
            cmap="RdYlGn",
            subset=["precision", "recall", "f1-score"]
        )
        .format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
        })
    )

    st.dataframe(styled_report, use_container_width=True)

    # --------------------------------------------------
    # CONFUSION MATRIX AND ROC CURVE
    # --------------------------------------------------
    
    st.markdown("### 📊 Confusion Matrix & ROC Curve")

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_test_enc, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_mapping.keys(),
            yticklabels=target_mapping.keys(),
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with col2:
        if y_prob is not None:
            classes = np.unique(y_test_enc)
            y_bin = label_binarize(y_test_enc, classes=classes)

            fig, ax = plt.subplots()
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, label=inverse_mapping[cls])

            ax.plot([0, 1], [0, 1], "--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves")
            ax.legend()
            st.pyplot(fig)

# --------------------------------------------------
# MODEL COMPARISON DASHBOARD
# --------------------------------------------------
st.header("📊 Model Comparison Dashboard")

compare_df = pd.DataFrame(comparison_results)

metric_map = {
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1 Score": "F1 Score",
    "MCC": "MCC",
    "ROC AUC": "AUC"
}

selected_metrics_labels = st.multiselect(
    "📌 Select metrics to compare",
    list(metric_map.keys()),
    default=["Accuracy", "F1 Score"]
)

selected_metrics = [metric_map[m] for m in selected_metrics_labels]

display_df = compare_df[["Model"] + selected_metrics]

st.dataframe(
    display_df.style
        .background_gradient(cmap="Blues", subset=selected_metrics)
        .format({m: "{:.4f}" for m in selected_metrics}),
    use_container_width=True
)

plot_df = display_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
plt.xticks(rotation=20)
plt.tight_layout()
st.pyplot(fig)
