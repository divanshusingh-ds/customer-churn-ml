# 🛡️ ChurnGuard AI

### Customer Churn Prediction & Risk Intelligence Platform

> **An end-to-end Machine Learning application that predicts customer churn, estimates churn risk, and provides explainable insights for customer retention.**

ChurnGuard AI is a complete **Customer Churn Prediction** project built using Python and Machine Learning. It combines data analysis, multiple classification models, hyperparameter tuning, Explainable AI, and an interactive Streamlit application into a single end-to-end solution.

The goal is simple:

**Identify customers who are likely to churn — understand why — and help businesses take action.**

---

## 🚀 Live Demo

🌐 **Streamlit App:** *Add your deployed Streamlit URL here*

📂 **GitHub Repository:** *This repository*

---

## 🎯 Business Problem

Customer churn is a major challenge for subscription-based businesses.

When a customer leaves, the company loses recurring revenue and may need to spend additional resources acquiring a replacement customer.

A predictive churn system can help businesses:

* Identify high-risk customers
* Prioritize retention campaigns
* Understand churn patterns
* Improve customer experience
* Allocate retention resources more effectively
* Take proactive action before customers leave

ChurnGuard AI transforms customer data into an actionable **risk intelligence system**.

---

# 🧠 How ChurnGuard AI Works

```text
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Categorical Encoding
      │
      ▼
Train / Test Split
      │
      ▼
Multiple ML Models
      │
      ▼
Model Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Best Model Selection
      │
      ├──────────────► Feature Importance
      │
      ├──────────────► SHAP Explainability
      │
      ▼
Saved ML Model
      │
      ▼
Streamlit Application
      │
      ▼
Churn Probability + Risk Level
      │
      ▼
Retention Recommendation
```

---

# 🤖 Machine Learning Models

Four classification algorithms were evaluated:

| Model               | Purpose                               |
| ------------------- | ------------------------------------- |
| Logistic Regression | Interpretable baseline classification |
| Decision Tree       | Rule-based classification             |
| Random Forest       | Ensemble learning                     |
| XGBoost             | Gradient boosting                     |

Random Forest and XGBoost were further optimized using **GridSearchCV**.

---

# 📊 Model Evaluation

Models were evaluated using multiple metrics instead of relying only on accuracy.

### Evaluation Metrics

* **Accuracy** — Overall prediction correctness
* **Precision** — How many predicted churners actually churned
* **Recall** — How many actual churners were identified
* **F1 Score** — Balance between precision and recall
* **ROC-AUC** — Overall ability to distinguish churners from non-churners
* **Confusion Matrix** — Detailed classification errors
* **ROC Curve** — Model discrimination across thresholds

### Model Comparison

The project stores the complete model comparison in:

```text
models/model_comparison.csv
```

The final model is selected based on **ROC-AUC performance after tuning**.

---

# 🔍 Explainable AI

A prediction is much more useful when we understand **why** the model made it.

ChurnGuard AI uses **SHAP (SHapley Additive exPlanations)** to analyze the contribution of features to model predictions.

This helps answer questions such as:

> Why is this customer considered high-risk?

> Which customer characteristics are influencing churn predictions?

> Which features are most important to the model?

The project also generates a feature-importance dataset:

```text
models/feature_importance.csv
```

---

# 🖥️ Interactive Streamlit Application

ChurnGuard AI includes an interactive web interface built with **Streamlit**.

Users can enter customer information including:

### 👤 Customer Profile

* Gender
* Senior citizen status
* Partner
* Dependents
* Tenure

### 📡 Services

* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming TV
* Streaming movies

### 💳 Billing

* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

The application then generates:

### 🔴 Churn Prediction

Whether the customer is predicted to churn.

### 📈 Churn Probability

A probability score representing the estimated likelihood of churn.

### 🚦 Risk Classification

```text
🟢 LOW RISK
🟠 MEDIUM RISK
🔴 HIGH RISK
```

### 💡 Retention Recommendation

The application translates the prediction into a business-oriented recommendation.

---

# 📁 Project Structure

```text
customer-churn-ml/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── confusion_matrix.json
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── roc_curve.csv
│
└── notebooks/
    └── 01_churn_analysis.ipynb
```

---

# 🧹 Data Preparation

The dataset was processed before model training.

Key preprocessing steps included:

* Duplicate detection and removal
* Removal of the customer ID field
* Conversion of `TotalCharges` to numeric format
* Missing-value handling
* Target encoding
* Categorical feature encoding
* Stratified train/test splitting

The preprocessing pipeline was integrated with the models using **Scikit-learn Pipelines and ColumnTransformer**.

This ensures that the same transformations are applied consistently during prediction.

---

# 📊 Exploratory Data Analysis

The analysis investigates relationships between customer characteristics and churn.

Important areas explored include:

* Overall churn distribution
* Contract type vs churn
* Tenure vs churn
* Monthly charges vs churn
* Internet service vs churn
* Payment method vs churn
* Technical support vs churn
* Feature correlations

These analyses provide the business context behind the machine learning problem.

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Model Management

* Joblib

### Application

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git & GitHub

---

# ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 📓 Run the Analysis

Open the Jupyter notebook:

```text
notebooks/01_churn_analysis.ipynb
```

The notebook contains the complete analytical workflow:

```text
Data Loading
      ↓
Data Inspection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Explainability
      ↓
Model Saving
```

---

# 💼 Business Impact

Churn prediction can support customer retention teams by helping them focus their efforts where they may have the greatest impact.

For example:

```text
High Churn Probability
        ↓
Identify Risk Factors
        ↓
Prioritize Customer
        ↓
Retention Strategy
        ↓
Potentially Reduce Churn
```

Possible retention strategies could include:

* Personalized offers
* Contract upgrades
* Customer support outreach
* Service improvements
* Targeted retention campaigns

> **Important:** ML predictions should support business decisions rather than replace human judgment.

---

# 📌 Key Project Highlights

### 🔹 End-to-End ML Pipeline

From raw customer data to a deployable prediction application.

### 🔹 Multiple Model Comparison

Four classification algorithms were evaluated.

### 🔹 Hyperparameter Optimization

Random Forest and XGBoost were optimized using GridSearchCV.

### 🔹 Explainable AI

SHAP and feature importance provide insight into model behavior.

### 🔹 Business-Oriented Predictions

Raw model output is converted into understandable customer risk levels.

### 🔹 Interactive Application

Users can test individual customer scenarios through Streamlit.

### 🔹 Reusable Preprocessing

Scikit-learn pipelines ensure preprocessing and prediction remain consistent.

---

# 📈 Future Improvements

Potential future versions could include:

* Real-time prediction API
* Cloud deployment
* Customer segmentation
* Automated retention campaign recommendations
* Model monitoring
* Drift detection
* Database integration
* Batch churn prediction
* Customer-level prediction history
* Advanced dashboard analytics

---

# 👨‍💻 Author

## Divanshu Singh

**Data Scientist | Machine Learning | Python | SQL | Power BI**

Interested in building practical systems that combine:

**Data → Intelligence → Automation → Business Impact**

---

# ⭐ If You Like This Project

If you found ChurnGuard AI interesting or useful, consider giving the repository a ⭐ on GitHub.

---

### 🛡️ ChurnGuard AI

**Predict. Explain. Act.**

*Turning customer data into actionable churn intelligence.*
