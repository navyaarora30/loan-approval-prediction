# 🏦 Loan Approval Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blue)![Status](https://img.shields.io/badge/Project-Completed-brightgreen)![Accuracy](https://img.shields.io/badge/Accuracy-82--83%25-success)

## 📌 Overview
This project builds a Machine Learning model to predict whether a loan application will be approved or rejected based on applicant financial and personal details.

The goal is to simulate a real-world loan approval system that is faster, consistent, and data-driven.

---

## 🏦 Problem Statement
A financial company processes hundreds of loan applications daily using manual verification.

This leads to:
- Time-consuming decision-making  
- Bias and inconsistency in approvals  
- Rejection of good customers  
- Approval of high-risk applicants  

To solve this, we build a Machine Learning system that:
- Analyzes applicant data automatically  
- Predicts loan approval (Approved/Rejected)  
- Assists decision-making before final human verification  

---

## 🎯 Business Objective
The objective of this project is to help financial institutions:
- Reduce loan processing time  
- Minimize human bias  
- Improve approval accuracy  
- Reduce financial risk  

---

## 🔍 What I Did
- Performed Exploratory Data Analysis (EDA) using Seaborn  
- Handled missing values and categorical encoding  
- Applied feature engineering (DTI ratio, credit score transformations)  
- Scaled features using StandardScaler  

---

## 🤖 Models Used
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  

---

## 📊 Results
- Best Model: Naive Bayes  
- Accuracy: ~82–83%  
- Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix  

---

## 💡 Key Insights
- Credit Score is the strongest predictor of loan approval  
- Higher Debt-to-Income (DTI) ratio reduces approval chances  
- Income alone is not sufficient without strong credit history  

---

## 📂 Dataset Features
The dataset includes applicant details such as:
- Applicant & Co-applicant Income  
- Credit Score  
- Employment Status  
- Debt-to-Income (DTI) Ratio  
- Loan Amount & Term  
- Savings & Collateral Value  
- Property Area & Education Level  

**Target Variable:**
- Loan_Approved (1 = Approved, 0 = Rejected)

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🚀 How to Run
bash
pip install -r requirements.txt
jupyter notebook loan_approval.ipynb

---

## 🚀 Author

**Navya Arora**  
BTech CSE Student | Data Science Intern @ Sabudh  

🔗 LinkedIn: http://linkedin.com/in/navya-arora-b19699252  
🔗 GitHub: https://github.com/navyaarora30  
