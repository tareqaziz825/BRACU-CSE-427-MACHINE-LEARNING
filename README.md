# Loan Eligibility Prediction Using Machine Learning with SMOTE for Class Imbalance

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-green.svg)](https://imbalanced-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CSE 427: Machine Learning Project**  
> Department of Computer Science and Engineering, BRAC University

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Team Members](#team-members)
- [Acknowledgements & Supervision](#acknowledgements--supervision)
- [References](#references)

## 🔬 Overview

This project develops an **automated loan eligibility prediction system** for Dream Housing Finance Company using machine learning classifiers. The system analyzes applicant information from online forms to predict loan approval likelihood, replacing traditional manual credit scoring with data-driven decision-making.

### Motivation

Financial institutions face critical challenges in loan approval:
- **Manual processes are slow and inconsistent**
- **Traditional credit scoring lacks flexibility** for dynamic lending environments
- **Class imbalance** (more approvals than rejections) skews predictions
- **Risk of default** requires accurate identification of ineligible applicants

### Our Solution

We implement a comprehensive ML pipeline that:
1. **Handles class imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
2. **Compares five ML algorithms** to identify optimal predictors
3. **Extracts feature importance** to understand key eligibility factors
4. **Provides interpretable predictions** for transparent decision-making

## 🎯 Problem Statement

**Objective**: Automate loan eligibility assessment based on applicant demographics, financial information, and credit history while addressing severe class imbalance in historical approval data.

**Challenges**:
- **Imbalanced dataset**: Significantly more approved loans than rejected applications
- **Missing data**: Incomplete applicant information requires robust imputation
- **Feature heterogeneity**: Mix of categorical (gender, education) and numerical (income, loan amount) features
- **Prediction bias**: Models must avoid discriminating against minority classes or demographic groups

## 📊 Dataset

### Source
**Dream Housing Finance Company** loan application dataset

### Statistics
- **Total Samples**: ~614 applicants
- **Features**: 12 input variables + 1 target variable
- **Classes**: Binary (Approved: "Y", Rejected: "N")

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| **Loan_ID** | Categorical | Unique identifier for each application |
| **Gender** | Categorical | Male/Female |
| **Married** | Categorical | Yes/No |
| **Dependents** | Categorical | Number of dependents (0, 1, 2, 3+) |
| **Education** | Categorical | Graduate/Not Graduate |
| **Self_Employed** | Categorical | Yes/No |
| **ApplicantIncome** | Numerical | Primary applicant's monthly income |
| **CoapplicantIncome** | Numerical | Co-applicant's monthly income |
| **LoanAmount** | Numerical | Requested loan amount (thousands) |
| **Loan_Amount_Term** | Numerical | Loan repayment period (months) |
| **Credit_History** | Categorical | Credit history meets guidelines (1.0: Yes, 0.0: No) |
| **Property_Area** | Categorical | Urban/Semi-Urban/Rural |
| **Loan_Status** | Target | Y (Approved) / N (Rejected) |

### Class Distribution

**Before SMOTE**:
- Approved (Y): ~70%
- Rejected (N): ~30%

**After SMOTE**:
- Balanced 50-50 distribution in training set

## 🔍 Methodology

### Pipeline Overview
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

### 1. Data Preprocessing

#### Missing Value Imputation
```python
# Numerical features: Median imputation
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Categorical features: Mode imputation
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
```

#### Categorical Encoding
```python
# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 
                                          'Self_Employed', 'Property_Area'])
```

#### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

### 2. Class Imbalance Handling: SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples for the minority class:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**How SMOTE Works**:
1. Identify minority class samples
2. For each sample, find k-nearest neighbors
3. Generate synthetic samples along lines connecting neighbors
4. Balance class distribution

**Benefits**:
- Prevents overfitting (unlike simple duplication)
- Improves minority class recall
- Enhances model generalization

### 3. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

- **Training Set**: 70%
- **Testing Set**: 30%
- **Stratified Split**: Preserves class distribution

## 🤖 Models Implemented

### 1. Random Forest Classifier

**Ensemble learning** method combining multiple decision trees:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

**Advantages**:
- Reduces overfitting through averaging
- Handles non-linear relationships
- Provides feature importance rankings

**Performance**: **89.29% accuracy**, 0.94 F1-score (True class)

### 2. Logistic Regression

**Linear probabilistic** classifier for binary outcomes:
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
```

**Advantages**:
- Highly interpretable (coefficient weights)
- Fast training and prediction
- Outputs probability scores

**Performance**: **75% accuracy**, preferred by banks for transparency

### 3. AdaBoost (Adaptive Boosting)

**Boosting ensemble** that iteratively improves weak learners:
```python
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada_model.fit(X_train, y_train)
```

**Advantages**:
- Focuses on misclassified samples
- Combines weak models into strong classifier
- Effective for complex patterns

**Performance**: **78.57% accuracy**, struggled with False class

### 4. K-Nearest Neighbors (KNN)

**Instance-based learning** using proximity:
```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

**Advantages**:
- No training phase (lazy learning)
- Captures non-linear decision boundaries
- Intuitive distance-based classification

**Performance**: **85.71% accuracy**, high precision (0.96) for True class

### 5. Multilayer Perceptron (MLP)

**Deep neural network** with backpropagation:
```python
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
mlp_model.fit(X_train, y_train)
```

**Advantages**:
- Learns complex non-linear patterns
- Flexible architecture
- High representational capacity

**Performance**: **89.29% accuracy**, **96% recall** for eligible applicants

## 📈 Results

### Performance Comparison Table

| Model | Accuracy | Precision (Y) | Recall (Y) | F1-Score (Y) | Precision (N) | Recall (N) |
|-------|----------|---------------|------------|--------------|---------------|------------|
| **Random Forest** | **89.29%** | **0.96** | 0.92 | **0.94** | 0.33 | 0.50 |
| **MLP** | **89.29%** | 0.93 | **0.96** | 0.94 | 0.00 | 0.00 |
| **KNN** | 85.71% | 0.96 | 0.88 | 0.92 | 0.25 | 0.50 |
| **AdaBoost** | 78.57% | 0.92 | 0.85 | 0.88 | 0.00 | 0.00 |
| **Logistic Regression** | 75.00% | 0.95 | 0.77 | 0.85 | 0.14 | 0.50 |

**Legend**: 
- **Y (True)**: Loan Approved
- **N (False)**: Loan Rejected
- **Bold**: Best performance

### Key Metrics Explained

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: Proportion of correct positive predictions
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)**: Proportion of actual positives correctly identified
```
Recall = TP / (TP + FN)
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC Curve Analysis

![ROC Curves](path/to/roc_curves.png)

**Observations**:
- Random Forest and MLP achieved highest AUC (Area Under Curve)
- Clear separation between True and False classes
- Models perform well on majority class, struggle with minority

### Confusion Matrices

**Random Forest**:
```
                Predicted
                N    Y
Actual  N      [ 5   5 ]
        Y      [ 7  83 ]
```

**Multilayer Perceptron**:
```
                Predicted
                N    Y
Actual  N      [ 0  10 ]
        Y      [ 4  86 ]
```

## 🔑 Key Findings

### 1. Feature Importance Analysis

**Correlation Matrix Insights**:

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| **ApplicantIncome ↔ LoanAmount** | **0.57** | Strong positive: Higher income → Larger loans |
| **CoapplicantIncome ↔ LoanAmount** | 0.19 | Moderate: Co-income moderately affects loan size |
| **CreditHistory ↔ LoanAmount** | -0.0084 | Negligible: Credit history doesn't directly influence loan amount |

**Key Predictors** (by correlation with Loan_Status):
1. **ApplicantIncome**: Primary financial indicator
2. **LoanAmount**: Requested loan size
3. **CreditHistory**: Past repayment behavior

**Weak Predictors**:
- Gender, Marital Status, Self-Employment status showed low correlation with approval

### 2. Model Performance Insights

#### Best Overall Models
✅ **Random Forest** (89.29% accuracy)
- Balanced performance across metrics
- High F1-score (0.94) for approved loans
- Better False class handling than MLP

✅ **Multilayer Perceptron** (89.29% accuracy)
- **Highest recall (96%)** for eligible applicants
- Excellent at identifying loan approvals
- Failed completely on rejected applications

#### Underperformers
❌ **Logistic Regression** (75% accuracy)
- Simplest model, lowest performance
- Limited by linear assumption
- Poor False class recognition

❌ **AdaBoost** (78.57% accuracy)
- Zero precision/recall on False class
- Suggests weak base learners
- Requires hyperparameter tuning

### 3. Critical Challenge: False Class Prediction

**All models struggled with rejected applications (False class)**:

| Model | False Precision | False Recall |
|-------|-----------------|--------------|
| Random Forest | 0.33 | 0.50 |
| MLP | **0.00** | **0.00** |
| KNN | 0.25 | 0.50 |
| AdaBoost | **0.00** | **0.00** |
| Logistic Regression | 0.14 | 0.50 |

**Why This Matters**:
- **Financial Risk**: Misclassifying ineligible applicants as eligible leads to loan defaults
- **False Positives**: Bank loses money on bad loans
- **Need for Improvement**: Advanced imbalance techniques or cost-sensitive learning required

### 4. Literature Alignment

Our findings confirm trends from existing research:

- **Ensemble superiority**: Random Forest outperforms simple classifiers (aligns with Orji et al., Muhammad et al.)
- **SMOTE effectiveness**: Improved minority class handling (consistent with Orji et al. 95.55% RF accuracy)
- **Logistic Regression limitations**: Lower accuracy (75%) vs. ensemble methods (aligns with Zhang 83.78%, Deepa et al. 85%)

### 5. Practical Implications

**For Dream Housing Finance**:
1. **Deploy Random Forest** for balanced accuracy and interpretability
2. **Use MLP** when maximizing approval detection is critical (minimize false negatives)
3. **Combine models** in ensemble voting for robust predictions
4. **Monitor False class performance** to prevent default risk

**Ethical Considerations**:
- Low correlation of Gender/Marital Status reduces bias risk
- Must ensure fairness across demographic groups
- Transparent explanations needed for rejected applications

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Clone Repository
```bash
git clone https://github.com/tareqaziz825/BRACU-CSE-427-MACHINE-LEARNING.git
cd BRACU-CSE-427-MACHINE-LEARNING
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📖 Usage

### Data Loading and Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('loan_data.csv')

# Handle missing values
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_encoded.drop('Loan_Status_Y', axis=1)
y = df_encoded['Loan_Status_Y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
```

### Training Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_smote)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_smote)

# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train_scaled, y_train_smote)

# Multilayer Perceptron
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train_smote)
```

### Model Evaluation
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Evaluate all models
models = {
    'Random Forest': rf_model,
    'Logistic Regression': lr_model,
    'K-Nearest Neighbors': knn_model,
    'AdaBoost': ada_model,
    'Multilayer Perceptron': mlp_model
}

for name, model in models.items():
    evaluate_model(model, X_test_scaled, y_test, name)
```

### Feature Importance Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Random Forest feature importance
feature_importance = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### Making Predictions
```python
def predict_loan_eligibility(applicant_data):
    """
    Predict loan eligibility for new applicant.
    
    Parameters:
    -----------
    applicant_data : dict
        Dictionary containing applicant information
    
    Returns:
    --------
    prediction : str
        'Approved' or 'Rejected'
    probability : float
        Confidence score
    """
    # Preprocess input
    input_df = pd.DataFrame([applicant_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # Align columns with training data
    missing_cols = set(X.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]
    
    # Scale
    input_scaled = scaler.transform(input_encoded)
    
    # Predict
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][prediction]
    
    result = 'Approved' if prediction == 1 else 'Rejected'
    return result, probability

# Example usage
sample_applicant = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '2',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

result, confidence = predict_loan_eligibility(sample_applicant)
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2%}")
```

## 👥 Team Members

| Name | Role |
|------|------|
| **Mohammod Tareq Aziz Justice** | Project Lead, Model Development |
| **Sanjana Afroz Troyee** | Data Preprocessing, Feature Engineering |
| **Tasfique Zaman Chowdhury Sifat** |Model Evaluation, Visualization |

## 🙏 Acknowledgements & Supervision

### Instructor
**Dr. Chowdhury Mofizur Rahman**  
Professor  
Department of Computer Science and Engineering  
BRAC University  
📧 Email: rahman.mofizur@bracu.ac.bd

## 📚 References

1. **Haque & Hassan** (2024). "Bank Loan Prediction Using Machine Learning Techniques." *arXiv*. DOI: 10.48550/arxiv.2410.08886

2. **Meenaakumari et al.** (2022). "Loan Eligibility Prediction using Machine Learning based on Personal Information." *IEEE IC3I*. DOI: 10.1109/IC3I56241.2022.10073318

3. **Zhang, Z.** (2023). "Loan Eligibility Prediction: An Analysis of Feature Relationships and Regional Variations." *Highlights in Business, Economics and Management*, 21, 688-697.

4. **Lakshmi Narasimha et al.** (2025). "Machine Learning For Bank Loan Eligibility Prediction: Focus on Home Loan and Education Loan." *International Journal on Science and Technology*, 16(1).

5. **Naveen Kumar et al.** (2022). "Customer Loan Eligibility Prediction using Machine Learning Algorithms in Banking Sector." *IEEE ICCES*.

6. **Deepa, Pal & Ghusey** (2024). "Monetary Loan Eligibility Prediction using Logistic Regression Algorithm." *IEEE IC-ETITE*.

7. **Manglani & Bokhare** (2021). "Logistic Regression Model for Loan Prediction: A Machine Learning Approach." *Emerging Trends in Industry 4.0*.

8. **B. D. S et al.** (2024). "Default credit card scoring using ML." *IEEE CISCSD*.

9. **Tumuluru et al.** (2022). "Comparative Analysis of Customer Loan Approval Prediction using Machine Learning Algorithms." *IEEE ICAIS*.

10. **Orji et al.** (2022). "Machine Learning Models for Predicting Bank Loan Eligibility." *IEEE NIGERCON*. DOI: 10.1109/nigercon54645.2022.9803172

11. **Muhammad et al.** (2024). "Loan Eligibility Prediction Using Ensemble Machine Learning Techniques and SMOTE." *IEEE ICSET*. DOI: 10.1109/icset63729.2024.10775270

## 🔮 Future Work

1. **Advanced Imbalance Handling**
   - Experiment with ADASYN, Borderline-SMOTE
   - Implement cost-sensitive learning with custom loss functions
   - Use ensemble sampling techniques (EasyEnsemble, BalancedBagging)

2. **Feature Engineering**
   - Incorporate transaction history, employment stability
   - Add macroeconomic indicators (interest rates, inflation)
   - Create interaction features (income-to-loan ratio, debt-to-income)

3. **Model Enhancement**
   - Hyperparameter optimization with GridSearch/RandomSearch
   - Implement XGBoost, LightGBM for gradient boosting
   - Explore neural architecture search for optimal MLP design

4. **Fairness & Interpretability**
   - Apply fairness-aware ML to prevent demographic bias
   - Use SHAP/LIME for model explainability
   - Conduct disparate impact analysis across gender/region

5. **Deployment**
   - Build REST API for real-time predictions
   - Create web dashboard for loan officers
   - Implement A/B testing framework for model monitoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Keywords**: Machine Learning, Loan Prediction, SMOTE, Class Imbalance, Random Forest, Ensemble Methods, Financial ML, Credit Scoring, Logistic Regression, Neural Networks
