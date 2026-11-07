import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classifier Platform", layout="wide")

st.title("🎯 Machine Learning Classifier Comparison Platform")
st.write("Upload test data, select models, and view model metrics and interactive evaluations instantly!")

# Sidebar options
st.sidebar.header("📁 Upload & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

model_mapping = {
    'Logistic Regression': 'model/logistic_regression_model.joblib',
    'Decision Tree': 'model/decision_tree_model.joblib',
    'KNN': 'model/knn_model.joblib',
    'Naive Bayes': 'model/naive_bayes_model.joblib',
    'Support Vector Machine': 'model/support_vector_machine_model.joblib'
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_mapping.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    
    with st.expander("🔍 Preview Test Data"):
        st.dataframe(test_df.head())
        
    if 'target' in test_df.columns:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Load the selected model
        model_path = model_mapping[selected_model_name]
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Inference
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display Metrics
            st.subheader(f"📊 Metrics Summary - {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix and Classification Report
            st.subheader("📈 Performance Visualization")
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("**Confusion Matrix**")
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
                
            with col_right:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().iloc[:-1, :-1])
        else:
            st.error("Model file not found. Please ensure training models are saved correctly.")
    else:
        st.error("The uploaded test CSV must contain the target column named 'target'.")
else:
    st.info("💡 Please upload a test CSV file containing 12 features and a 'target' column to see model results.")
