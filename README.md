# 🏦 Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be **approved or rejected** based on customer details such as income, credit history, loan amount, and employment status. The dataset used in this project is the **Loan Prediction Problem Dataset** (popular on Kaggle).

---

## 🚀 Project Goal
To build a machine learning model that can **automate the loan approval process** and assist financial institutions in making faster and more reliable decisions.

---

## 📂 Dataset Information
The dataset contains information about loan applicants:

| Feature | Description |
|---------|-------------|
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Loan amount requested |
| Loan_Amount_Term | Duration of loan |
| Credit_History | Previous history of loan repayment |
| Property_Area | Urban / Rural / Semiurban |
| Loan_Status | Target variable → Y (Approved) / N (Rejected) |

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:

- Handling missing values (median for numerical features, mode for categorical features)
- Label encoding for categorical features
- Train–test split (80% training, 20% testing)
- Standardization of numerical features (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`)
- Outlier handling and feature scaling

---

## 🤖 Machine Learning Models Used
| Model | Accuracy |
|--------|----------|
| **Logistic Regression** | **79.67%** |
| Decision Tree Classifier | 71% |
| K-Nearest Neighbors (KNN) | 69% |

➡ **Logistic Regression achieved the best performance and was chosen as the final model.**

---

## 🔧 Hyperparameter Tuning
A `GridSearchCV` pipeline was applied to improve accuracy by tuning:

- `C`
- `solver`
- `penalty`
- `max_iter`

The best parameters were selected based on cross-validation accuracy.

---

## 🏆 Final Model Saving
The final trained model was saved using `joblib`:

```python
joblib.dump(lr, "loan_prediction.pkl")
