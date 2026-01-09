import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="ML Classification Model Evaluator", layout="wide")
st.title("📊 ML Classification Model Evaluator")

st.write("Upload **TEST dataset only (CSV)**")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_map = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    
    # Load target class mapping CSV and encode target
    mapping_df = pd.read_csv("model/target_class_encoding.csv")
    target_mapping = dict(zip(mapping_df['class'], mapping_df['encoded']))
    y_test_encoded = y_test.map(target_mapping).values
    st.write("✓ Test data and target variable loaded successfully.")

    with open(f"model/{model_map[model_choice]}", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        st.write(f"Prediction Probabilities: {y_prob}")
        auc = roc_auc_score(y_test_encoded, y_prob, multi_class='ovr')
    except:
        auc = "N/A"

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", accuracy_score(y_test_encoded, y_pred))
    col1.metric("Precision", precision_score(y_test_encoded, y_pred, zero_division=0, average="weighted"))
    col1.metric("Recall", recall_score(y_test_encoded, y_pred, zero_division=0, average="weighted"))

    col2.metric("F1 Score", f1_score(y_test_encoded, y_pred, zero_division=0, average="weighted"))
    col2.metric("MCC", matthews_corrcoef(y_test_encoded, y_pred))
    col2.metric("AUC", auc)

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test_encoded, y_pred))
