import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import zipfile
import io

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

st.markdown("""
    <style>
    .stApp {
        color: #3f3a2c;
    }

    section[data-testid="stSidebar"] {
        background-color: #f2f1ea;
        border-right: 1px solid #e3e1d9;
    }

    h1, h2, h3, h4 {
        color: #3f3a2c;
        font-weight: 600;
    }

    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"],
    div[data-testid="stExpander"],
    div[data-testid="stContainer"] {
        background-color: #f6f5ee;
        border: 1px solid #e6e4db;
        border-radius: 12px;
        padding: 12px;
    }
            
    div[data-testid="stRadio"],
    div[data-testid="stSelectbox"],
    div[data-testid="stMultiSelect"] {
        background-color: #f6f5ee;
        border: 1px solid #e0ddd2;
        border-radius: 12px;
        padding: 10px;
    }

    button[kind="primary"] {
        background-color: #3f3a2c;
        color: #ffffff;
        border-radius: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        width: 100% !important;
    }

    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
    }

    div[data-testid="stDataFrame"] table {
        width: 100% !important;
    }            
            
    span[data-baseweb="tag"] {
        background-color: #d5a655 !important;
    }

    </style>
""", unsafe_allow_html=True)


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
st.sidebar.header("⚙️ View Options")
show_dataset_info = st.sidebar.toggle(
    "Show Dataset Info",
    value=True,
    help="Toggle dataset description for evaluators"
)

st.sidebar.header("📂 Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("— or —")
st.sidebar.caption("Use this sample if you don’t have test data ready")

sample_df = pd.read_csv("data/test_data.csv")

st.sidebar.download_button(
    label="⬇️ Download Sample Test Data",
    data=sample_df.to_csv(index=False),
    file_name="sample_test_data.csv",
    mime="text/csv"
)

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------
if show_dataset_info:
    data = pd.read_csv("data/train_data.csv")
    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    st.header("📘 Dataset Information")
    with st.expander("📚 Predict Students' Dropout and Academic Success", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            **Dataset Source:** UCI Machine Learning Repository  

            This dataset focuses on predicting **student academic outcomes**
            using demographic, socioeconomic, and academic performance data.

            **Objective:**  
            Early identification of students at risk of **dropping out** to
            enable timely academic intervention.

            **ML Task:** Multi-class Classification
            """)
        with col2:
            st.metric("Total Students", data.shape[0])
            st.metric("Input Features", data.shape[1] - 1)
            st.metric("Target Classes", y_train.nunique())
            st.metric("Missing Values", int(data.isnull().sum().sum()))

    with st.expander("📊 Target Class Distribution"):
        class_counts = (
            y_train.value_counts()
            .reset_index()
            .rename(columns={"index": "Class", y_train.name: "Count"})
        )
        fig = px.bar(
            class_counts,
            x="Class",
            y="Count",
            text="Count",
            title="Student Academic Outcome Distribution"
        )
        fig.update_layout(
            height=400,
            yaxis_title="Number of Students",
            xaxis_title="Academic Outcome"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            class_counts
            .style
            .background_gradient(cmap="Greens", subset=["Count"]),
            use_container_width=True
        )

    with st.expander("🧬 Feature Groups (36 Input Features)"):
        def feature_table(title, features):
            df = pd.DataFrame({"Feature Name": features})
            st.subheader(title)
            st.dataframe(
                df.style
                .set_properties(**{"text-align": "left"})
                .background_gradient(cmap="Greens"),
                use_container_width=True
            )
        feature_table("👤 Demographic Features", [
            "Age at enrollment",
            "Gender",
            "Marital status",
            "Nationality"
        ])
        feature_table("🏫 Academic Background", [
            "Course",
            "Previous qualification",
            "Admission grade",
            "Daytime/evening attendance"
        ])
        feature_table("👨‍👩‍👧 Socioeconomic Factors", [
            "Mother qualification",
            "Father qualification",
            "Mother occupation",
            "Father occupation",
            "Scholarship holder",
            "Tuition fees up to date"
        ])
        feature_table("📊 Academic Performance", [
            "Curricular units 1st semester (credited)",
            "Curricular units 1st semester (approved)",
            "Curricular units 1st semester (grade)",
            "Curricular units 2nd semester (credited)",
            "Curricular units 2nd semester (approved)",
            "Curricular units 2nd semester (grade)"
        ])

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
st.sidebar.header("🎰 Model Selection")

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
classification_reports = {}

# --------------------------------------------------
# MODEL EVALUATION LOOP
# --------------------------------------------------
for model_name in selected_models:
    st.subheader(f"🧮 {model_name}")
    st.info(MODEL_INFO[model_name][1])

    with open(f"model/{MODEL_INFO[model_name][0]}", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted")
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
    st.markdown("### 📋 Classification Report")

    report = classification_report(
        y_test_enc, y_pred,
        target_names=target_mapping.keys(),
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(index="accuracy", errors="ignore")
    csv_report = (
        report_df
        .reset_index()
        .rename(columns={"index": "Class"})
    )
    classification_reports[model_name] = csv_report

    # --------------------------------------------------
    # DOWNLOAD CLASSIFICATION REPORTS
    # --------------------------------------------------
    csv_report = report_df.reset_index().rename(columns={"index": "Class"})

    st.download_button(
        label=f"⬇️ Download {model_name} Classification Report",
        data=csv_report.to_csv(index=False),
        file_name=f"{model_name.lower().replace(' ', '_')}_classification_report.csv",
        mime="text/csv"
    )

    # --------------------------------------------------
    # STYLED CLASSIFICATION REPORT
    # --------------------------------------------------

    styled_report = (
        report_df
        .style
        .background_gradient(
            cmap="Greens",
            subset=["precision", "recall", "f1-score", "support"]
        )
        .format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
            "support": "{:.0f}"
        })
    )

    st.dataframe(styled_report, width="stretch")

    # --------------------------------------------------
    # CONFUSION MATRIX & ROC CURVE (PLOTLY)
    # --------------------------------------------------
    st.markdown("### 📊 Confusion Matrix & ROC Curve")

    col1, col2 = st.columns(2)

    # ---------- Confusion Matrix (Plotly) ----------
    with col1:
        cm = confusion_matrix(y_test_enc, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=target_mapping.keys(),
            columns=target_mapping.keys()
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="YlGnBu",
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            title="Confusion Matrix"
        )

        fig_cm.update_layout(
            height=420,
            margin=dict(l=40, r=40, t=60, b=40),
            coloraxis_colorbar=dict(title="Samples")
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    # ---------- ROC Curve (Plotly) ----------
    with col2:
        if y_prob is not None:
            classes = np.unique(y_test_enc)
            y_bin = label_binarize(y_test_enc, classes=classes)

            roc_data = []

            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_data.append(
                    pd.DataFrame({
                        "False Positive Rate": fpr,
                        "True Positive Rate": tpr,
                        "Class": inverse_mapping[cls]
                    })
                )

            roc_df = pd.concat(roc_data)

            fig_roc = px.line(
                roc_df,
                x="False Positive Rate",
                y="True Positive Rate",
                color="Class",
                title="ROC Curves (One-vs-Rest)"
            )

            # Diagonal reference line
            fig_roc.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=0, y0=0, x1=1, y1=1
            )

            fig_roc.update_layout(
                height=420,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend_title_text="Class",
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve not available for this model.")

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
        .background_gradient(cmap="Greens", subset=selected_metrics)
        .format({m: "{:.4f}" for m in selected_metrics}),
    width="stretch"
)

plot_df = display_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    plot_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".4f"
)

fig.update_layout(
    title="Model Comparison",
    xaxis_title="Model",
    yaxis_title="Score",
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=500
)

st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# SIDEBAR – DOWNLOAD TEST REPORTS
# --------------------------------------------------
with st.sidebar.expander("📥 Download Test Reports"):

    # ---------- Model Metrics Report ----------
    comparison_df = pd.DataFrame(comparison_results)

    st.download_button(
        label="⬇️ Model Metrics Report (CSV)",
        data=comparison_df.to_csv(index=False),
        file_name="model_metrics_report.csv",
        mime="text/csv",
        help="Download accuracy, precision, recall, F1, MCC, and AUC for all models"
    )

    # ---------- Selected Metrics Comparison ----------
    st.download_button(
        label="⬇️ Model Comparison Table (CSV)",
        data=display_df.to_csv(index=False),
        file_name="model_comparison_selected_metrics.csv",
        mime="text/csv",
        help="Download the selected comparison metrics"
    )

    # ---------- Full Evaluation ZIP (Optional) ----------
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(
            "model_metrics_report.csv",
            comparison_df.to_csv(index=False)
        )
        zipf.writestr(
            "model_comparison_selected_metrics.csv",
            display_df.to_csv(index=False)
        )
        # Per-model classification reports
        for model_name, report_df in classification_reports.items():
            file_name = (
                f"classification_reports/"
                f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
            )

            zipf.writestr(
                file_name,
                report_df.to_csv(index=False)
            )
    zip_buffer.seek(0)
    st.download_button(
        label="📦 Full Evaluation Report (ZIP)",
        data=zip_buffer,
        file_name="ml_model_evaluation_reports.zip",
        mime="application/zip",
        help="Download all evaluation reports in a single ZIP"
    )

# --------------------------------------------------
# END OF APP
# --------------------------------------------------
