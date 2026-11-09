# Student Performance Prediction System

## Project Description
The Student Performance Prediction System is a Data Mining classroom prototype that uses machine learning to identify students who may be academically at risk based on historical student information.

## Problem Statement
Academic difficulties are sometimes identified only after students receive poor academic results. This system explores how historical student data can help identify learners who may need additional academic support.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Student Performance
- Subject: Mathematics
- Records: 649 students
- Target variable: G3 (Final Grade)

## Prediction Rule
- G3 < 10 = AT RISK
- G3 >= 10 = NOT AT RISK

G3 is excluded from the model's input features to prevent target leakage.

## Technologies Used
- Python
- Google Colab
- Visual Studio Code
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- GitHub

## Model Evaluation
The Random Forest model achieved:
- Accuracy: 89.2%
- At-risk precision: 75%
- At-risk recall: 45%
- At-risk F1-score: 56%

## Features
- Dataset overview and preview
- Student CSV upload
- Required-column validation
- Academic risk prediction
- Prediction results display
- Downloadable prediction results

## Installation and Execution
Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations
This model uses historical data from Portuguese secondary schools. Its predictions may not generalize to other institutions or student populations. Predictions are for classroom demonstration only and should not be used as the sole basis for real student decisions.

## Reference
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/320/student+performance