import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load model, vectorizer, and test data using Streamlit's new caching system
@st.cache_data
def load_resources():
    with open('profanity_logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)

    return model, vectorizer, X_test, y_test

# Load all resources (model, vectorizer, test data)
model, vectorizer, X_test, y_test = load_resources()

# Function to predict whether a comment is profane or not
def predict_comment(comment):
    comment_vectorized = vectorizer.transform([comment])
    prediction = model.predict(comment_vectorized)
    return prediction[0]

# Streamlit Interface
st.title("Profanity Comment Classifier")

# Function to handle comment submission
def handle_submit():
    if st.session_state.comment:
        prediction = predict_comment(st.session_state.comment)
        st.session_state.prediction = prediction

# Text input for user to enter comment with 'Enter' key triggering submit
st.text_input("Enter Comment Below:", key="comment", on_change=handle_submit)

# Manual submit button in case user clicks instead of pressing 'Enter'
if st.button("Submit"):
    handle_submit()

# Show the prediction result after the submit button
if 'prediction' in st.session_state:
    prediction = st.session_state.prediction
    if prediction == 1:
        st.markdown('<p style="color:red; font-size:24px;">Prediction: Profane</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size:24px;">Prediction: Non-Profane</p>', unsafe_allow_html=True)

# Sidebar with options for visualizations
st.sidebar.header("Options")

# Functions to plot graphs (included from your original code)
def show_accuracy():
    y_pred = model.predict(X_test)
    y_test_clean = np.nan_to_num(y_test, nan=0).astype(int)
    y_pred_clean = np.nan_to_num(y_pred, nan=0).astype(int)
    acc = accuracy_score(y_test_clean, y_pred_clean) * 100
    st.info(f"Accuracy: {acc:.2f}%")

def plot_distribution_by_number():
    y_pred = model.predict(X_test)
    y_test_clean = np.nan_to_num(y_test, nan=0).astype(int)
    y_pred_clean = np.nan_to_num(y_pred, nan=0).astype(int)
    cm = confusion_matrix(y_test_clean, y_pred_clean)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Profane", "Profane"],
                yticklabels=["Non-Profane", "Profane"], ax=ax)
    ax.set_title('Profanity Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def plot_category_percentage():
    labels = ['Non-Profane', 'Profane']
    counts = [sum(y_test == 0), sum(y_test == 1)]
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    ax.set_title('Profanity Category Distribution')
    st.pyplot(fig)

def plot_heatmap_of_metrics():
    y_pred = model.predict(X_test)
    y_test_clean = np.nan_to_num(y_test, nan=0).astype(int)
    y_pred_clean = np.nan_to_num(y_pred, nan=0).astype(int)
    report = classification_report(y_test_clean, y_pred_clean, labels=[0, 1], target_names=["Non-Profane", "Profane"], output_dict=True)
    precision = [report["Non-Profane"]["precision"], report["Profane"]["precision"]]
    recall = [report["Non-Profane"]["recall"], report["Profane"]["recall"]]
    f1_score = [report["Non-Profane"]["f1-score"], report["Profane"]["f1-score"]]
    metrics = np.array([precision, recall, f1_score])
    fig, ax = plt.subplots()
    sns.heatmap(metrics, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=["Precision", "Recall", "F1-Score"],
                yticklabels=["Non-Profane", "Profane"], ax=ax)
    ax.set_title("Heatmap of Precision, Recall, and F1 Scores")
    st.pyplot(fig)

# Add buttons for displaying visualizations
if st.sidebar.button("Show Prediction Accuracy"):
    show_accuracy()

if st.sidebar.button("Show Profanity Matrix"):
    plot_distribution_by_number()

if st.sidebar.button("% of Comments"):
    plot_category_percentage()

if st.sidebar.button("Show Heatmap"):
    plot_heatmap_of_metrics()
