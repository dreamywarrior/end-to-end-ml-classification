import streamlit as st
import pandas as pd
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Log the loading of the test data
    logging.info('Loading test data...')
    data = pd.read_csv(uploaded_file)

    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    
    # Load target class mapping CSV and encode target
    logging.info('Encoding target variable...')
    mapping_df = pd.read_csv("model/target_class_encoding.csv")
    target_mapping = dict(zip(mapping_df['class'], mapping_df['encoded']))
    y_test_encoded = y_test.map(target_mapping).values
    st.write("✓ Test data and target variable loaded successfully.")

    with open(f"model/{model_map[model_choice]}", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test_encoded, y_prob, multi_class='ovr')
    except:
        auc = "N/A"

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, zero_division=0, average="weighted")
    recall = recall_score(y_test_encoded, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(y_test_encoded, y_pred, zero_division=0, average="weighted")

    # Log metrics with 5 decimal places
    logging.info(f'Accuracy: {accuracy:.5f}')
    logging.info(f'Precision: {precision:.5f}')
    logging.info(f'Recall: {recall:.5f}')
    logging.info(f'F1 Score: {f1:.5f}')

    col1.metric("Accuracy", round(accuracy, 5))
    col1.metric("Precision", round(precision, 5))
    col1.metric("Recall", round(recall, 5))

    col2.metric("F1 Score", round(f1, 5))
    col2.metric("MCC", round(matthews_corrcoef(y_test_encoded, y_pred), 5))
    col2.metric("AUC", auc)

    st.subheader("📉 Confusion Matrix")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=list(target_mapping.keys()),
                yticklabels=list(target_mapping.keys()))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
