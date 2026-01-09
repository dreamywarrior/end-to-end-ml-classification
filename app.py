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
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    
    # Load label encoder and encode target
    label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
    y_test_encoded = label_encoder.transform(y_test)

    with open(f"model/{model_map[model_choice]}", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_encoded, y_prob)
    except:
        auc = "N/A"

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", accuracy_score(y_test_encoded, y_pred))
    col1.metric("Precision", precision_score(y_test_encoded, y_pred, zero_division=0))
    col1.metric("Recall", recall_score(y_test_encoded, y_pred, zero_division=0))

    col2.metric("F1 Score", f1_score(y_test_encoded, y_pred, zero_division=0))
    col2.metric("MCC", matthews_corrcoef(y_test_encoded, y_pred))
    col2.metric("AUC", auc)

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
