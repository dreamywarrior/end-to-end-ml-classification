# 🎓 Predict Students' Dropout and Academic Success

## Problem Statement

This project implements and compares six different machine learning classification models to predict students' academic outcomes — Dropout, Enrolled, or Graduate. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and deployment through an interactive Streamlit web application.

The assignment requires:

- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Predict Students' Dropout and Academic Success from UCI Machine Learning Repository

**Source:** [UCI ML Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

**Description:**
The dataset was created using data from a higher education institution, collected from multiple disjoint databases. It focuses on students enrolled in various undergraduate programs, including Agronomy, Design, Education, Nursing, Journalism, Management, Social Service, and Technologies. The dataset contains information available at the time of enrollment, along with academic performance at the end of the first and second semesters.

**Features (36 input features):**
The dataset includes demographic, socioeconomic, and academic performance features such as:

- Marital status, Course, Daytime/evening attendance
- Previous qualification, Nationality, Mother's/Father's qualification and occupation
- Admission grade, Age at enrollment, Curricular units credits
- Grade point averages for 1st and 2nd semester
- And many more academic and personal attributes

**Target Variable:**

- **Target** - Student status: Dropout, Enrolled, Graduate (3 classes)

**Dataset Statistics:**

- **Total Instances:** 4,424
- **Features:** 36
- **Classes:** 3 (Dropout, Enrolled, Graduate)
- **Missing Values:** None

## Models Used

The following **six classification models** were implemented and evaluated using the same dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble Model)**
6. **XGBoost (Ensemble Model)**

### Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **AUC Score**: Area Under the ROC Curve (one-vs-rest for multi-class)
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **MCC**: Matthews Correlation Coefficient (balanced measure for binary/multi-class)

### Comparison Table with Evaluation Metrics

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.7480   | 0.8887  | 0.7290    | 0.7480  | 0.7321   | 0.5858  |
| Decision Tree        | 0.6814   | 0.7529  | 0.6797    | 0.6814  | 0.6800   | 0.4844  |
| KNN                  | 0.6090   | 0.7229  | 0.5973    | 0.6090  | 0.5977   | 0.3537  |
| Naive Bayes          | 0.6994   | 0.8240  | 0.6877    | 0.6994  | 0.6868   | 0.5056  |
| Random Forest        | 0.7605   | 0.8870  | 0.7451    | 0.7605  | 0.7446   | 0.6071  |
| XGBoost              | 0.7638   | 0.8940  | 0.7541    | 0.7638  | 0.7553   | 0.6139  |

### Observations about Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieves **strong performance** with an accuracy of **0.7480** and a high **AUC of 0.8887**, indicating good class separability. The balanced **precision (0.7290)** and **recall (0.7480)** lead to a solid **F1 score (0.7321)**, showing consistent predictions across classes. This performance suggests that many academic and demographic features have approximately linear relationships with student outcomes, making Logistic Regression a reliable baseline model. |
| Decision Tree | Decision Tree attains **moderate performance** with **accuracy of 0.6814** and **AUC of 0.7529**. While its **precision (0.6797)** and **recall (0.6814)** are comparable, the lower **F1 score (0.6800)** and **MCC (0.4844)** indicate weaker generalization. This is mainly due to **overfitting**, as a single tree struggles with high-dimensional feature space and class imbalance present in the dataset. |
| KNN | KNN shows the **lowest overall performance**, with **accuracy of 0.6090**, **AUC of 0.7229**, and **MCC of 0.3537**. The relatively low **precision (0.5973)** and **F1 score (0.5977)** indicate inconsistent neighborhood-based predictions. This behavior is expected due to the **curse of dimensionality** (36 features) and sensitivity to feature scaling, which reduces the effectiveness of distance-based learning in this dataset. |
| Naive Bayes | Naive Bayes achieves **reasonable performance** with **accuracy of 0.6994** and a comparatively high **AUC of 0.8240**, indicating good probabilistic ranking. However, the modest **precision (0.6877)** and **F1 score (0.6868)** suggest some misclassification at the decision boundary. This occurs because the model’s **feature independence assumption** does not fully hold for correlated academic and socioeconomic variables. |
| Random Forest | Random Forest delivers **strong and stable performance**, achieving **accuracy of 0.7605** and **AUC of 0.8870**. The high **precision (0.7451)** and **recall (0.7605)** result in a strong **F1 score (0.7446)** and **MCC (0.6071)**. The ensemble of trees effectively reduces overfitting and captures complex, non-linear relationships among features, making it well-suited for this dataset. |
| XGBoost | XGBoost achieves the **best overall performance** with the highest **accuracy (0.7638)**, **AUC (0.8940)**, **F1 score (0.7553)**, and **MCC (0.6139)**. Its strong **precision (0.7541)** and **recall (0.7638)** indicate an excellent balance between false positives and false negatives. The gradient boosting approach allows the model to iteratively correct errors and handle class imbalance effectively, resulting in superior predictive performance. |

## Project Structure

```
end-to-end-ml-classification/
├── app.py                          # Main Streamlit application
├── model_training.ipynb            # Jupyter notebook for model training
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Directory for datasets
│   ├── train_data.csv              # Training dataset for the models
│   └── test_data.csv               # Test dataset for evaluation
├── model/                          # Directory for saved models
│    ├── *.pkl                      # Trained model files
│    └── target_class_encoding.csv  # Class encoding mapping
└── resources/                      # Directory for image resources for app
     └── model_evaluator.jpg        # Model evaluation workflow iamge
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd end-to-end-ml-classification
   ```

2. **Prepare the train and test dataset**

   ```bash
   jupyter notebook prepare_dataset.ipynb
   ```

3. **Train the models based on training data**

   ```bash
   jupyter notebook model_training.ipynb
   ```

4. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

## Streamlit Application Features

1. **Model Evaluation Page**

   - Upload test CSV data
   - Select one or multiple models for comparison
   - View detailed metrics for each model
   - Compare confusion matrices across models
   - Interactive dashboard with performance visualizations

2. **Model Comparison Dashboard**
   - Side-by-side comparison of all selected models
   - Customizable metric selection
   - Bar charts for visual comparison
   - Formatted results table with color coding

## Deployment

### Streamlit Community Cloud

**Quick Deployment Steps:**

1. **Push your code to GitHub** (already done ✅)

   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to Streamlit Cloud**

   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App**

   - Click "New App" button
   - Select your repository
   - Set main file path to: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Deployment takes 2-5 minutes
   - Your app will be live at: `https://<your-app-name>.streamlit.app`

### Automatic Updates

- Streamlit Cloud automatically redeploys when you push to the `main` branch
- Just push your changes: `git push origin main`
- Wait 2-5 minutes for automatic redeployment
