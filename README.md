# Wine Quality Prediction using RandomForest

## Project Overview

This project predicts whether a wine is **good quality or bad quality** based on its physicochemical properties.

The goal of the project was to build a complete machine learning workflow including:

- data exploration
- model training
- hyperparameter tuning
- evaluation
- building an API
- creating a simple web interface

The final model is deployed through a **FastAPI backend** and a **Streamlit interface**.

---

# Dataset

The dataset used is the **WineQT dataset**, which contains chemical properties of wine samples.

Features include:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

The target variable **quality** was converted into binary classification:
quality ≥ 7 → Good Wine
quality < 7 → Bad Wine


---

# Exploratory Data Analysis

Some basic analysis was performed to understand the data:

- distribution of wine quality
- relationship between quality and chemical properties
- correlation heatmap between features

These visualizations helped understand which features may influence wine quality.

---

# Model

The model used for prediction is **Random Forest Classifier**.

Model improvements included:

- hyperparameter tuning using **GridSearchCV**
- evaluation using **precision, recall, F1 score, and ROC-AUC**

---

# Results

Model performance on the test set:

Accuracy: ~93%

Confusion Matrix:

---

# Exploratory Data Analysis

Some basic analysis was performed to understand the data:

- distribution of wine quality
- relationship between quality and chemical properties
- correlation heatmap between features

These visualizations helped understand which features may influence wine quality.

---

# Model

The model used for prediction is **Random Forest Classifier**.

Model improvements included:

- hyperparameter tuning using **GridSearchCV**
- evaluation using **precision, recall, F1 score, and ROC-AUC**

---

# Results

Model performance on the test set:

Accuracy: ~93%

Confusion Matrix:
[[194 7]
[ 9 19]]


Evaluation metrics:

Precision: ~0.73  
Recall: ~0.68  
F1 Score: ~0.70  
ROC-AUC: ~0.96

The model performs well overall while still capturing the minority class reasonably.

---

# Project Structure


wine-quality-prediction
│
├── train_model.py
├── app.py
├── streamlit_app.py
├── wine_model.pkl
├── WineQT.csv
└── README.md


---

# Running the Project

### Train the model


python train_model.py


### Run the API


uvicorn app:app --reload


API documentation:


http://127.0.0.1:8000/docs


### Run the web interface


streamlit run streamlit_app.py


The interface will open at:


http://localhost:8501


---

# Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
FastAPI  
Streamlit

---

# Author

Harshini Gondesi
