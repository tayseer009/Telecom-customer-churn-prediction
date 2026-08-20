# 📊 Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project covers the complete machine learning workflow, including exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, threshold optimization, error analysis, final test evaluation, and SHAP-based model explainability.

---

## 🎯 Project Overview

Customer churn is an important business problem for telecom companies because retaining existing customers is generally more valuable than constantly acquiring new ones.

The objective of this project is to build a machine learning model that can identify customers who are likely to churn so that the business can take proactive retention actions.

The project focuses not only on achieving high accuracy, but also on improving the model's ability to identify actual churners.

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning pipeline:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Train / Test Split
     ↓
Preprocessing Pipeline
     ↓
Baseline Models
     ↓
5-Fold Cross-Validation
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Final Model Selection
     ↓
Threshold Optimization
     ↓
Error Analysis
     ↓
Final Test Evaluation
     ↓
SHAP Explainability
     ↓
Business Interpretation
```

---

## 📁 Project Structure

```text
Customer Churn Prediction/
│
├── images/
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── threshold_analysis.png
│
├── telecom_churn_prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

> The raw dataset is excluded from the repository through `.gitignore`.

---

## 🔎 Exploratory Data Analysis

The project begins with exploratory analysis to understand:

* Customer churn distribution
* Feature distributions
* Relationships between customer characteristics and churn
* Potential data quality issues
* Features that may contribute to customer churn

### Churn Distribution

![Churn Distribution](images/churn_distribution.png)

---

## ⚙️ Data Preprocessing

The preprocessing stage includes the appropriate transformations for the dataset's numerical and categorical features.

The project uses a preprocessing pipeline to ensure that transformations are performed consistently and to reduce the risk of data leakage during cross-validation.

The workflow includes:

* Missing-value handling
* Numerical feature preprocessing
* Categorical feature encoding
* Feature scaling where required
* Scikit-learn pipelines

---

## 🤖 Models Evaluated

Several traditional machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Gradient Boosting

The models were evaluated using **5-fold cross-validation**.

The main evaluation metrics were:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

ROC-AUC was particularly important during model selection because the project focuses on identifying customers who are likely to churn.

---

## 🔄 Cross-Validation Results

### Logistic Regression

```text
Accuracy : 0.8021 ± 0.0116
Precision: 0.6529 ± 0.0255
Recall   : 0.5431 ± 0.0411
F1 Score : 0.5923 ± 0.0296
ROC-AUC  : 0.8461 ± 0.0125
```

### Decision Tree

```text
Accuracy : 0.7302 ± 0.0096
Precision: 0.4919 ± 0.0177
Recall   : 0.5057 ± 0.0211
F1 Score : 0.4986 ± 0.0181
ROC-AUC  : 0.6583 ± 0.0124
```

### Random Forest

```text
Accuracy : 0.7856 ± 0.0124
Precision: 0.6242 ± 0.0333
Recall   : 0.4849 ± 0.0233
F1 Score : 0.5455 ± 0.0243
ROC-AUC  : 0.8202 ± 0.0128
```

### SVM

```text
Accuracy : 0.8035 ± 0.0088
Precision: 0.6772 ± 0.0276
Recall   : 0.4977 ± 0.0157
F1 Score : 0.5735 ± 0.0162
ROC-AUC  : 0.7998 ± 0.0160
```

### Gradient Boosting

```text
Accuracy : 0.8033 ± 0.0110
Precision: 0.6606 ± 0.0289
Recall   : 0.5338 ± 0.0234
F1 Score : 0.5902 ± 0.0227
ROC-AUC  : 0.8478 ± 0.0122
```

---

## 🏆 Model Selection

After cross-validation and hyperparameter tuning, the strongest models were:

| Model                 | Tuned ROC-AUC |
| --------------------- | ------------: |
| **Gradient Boosting** |    **0.8509** |
| Logistic Regression   |        0.8463 |

Gradient Boosting was selected as the final model because it achieved the highest tuned ROC-AUC.

---

## 🎚️ Threshold Optimization

Instead of automatically using the standard classification threshold of `0.50`, the project evaluated different probability thresholds.

The best threshold based on F1 Score was:

```text
Best Threshold: 0.34
```

At this threshold:

```text
Accuracy : 0.7879
Precision: 0.5801
Recall   : 0.7264
F1 Score : 0.6451
```

The lower threshold increases the model's ability to identify customers who are actually going to churn.

### Threshold Analysis

![Threshold Analysis](images/threshold_analysis.png)

---

## 🔍 Error Analysis

The model's predictions were analyzed to understand where it makes mistakes.

Using the selected threshold:

```text
Correct predictions : 4439 (78.79%)
False positives     : 786  (13.95%)
False negatives     : 409  (7.26%)
```

Confusion matrix:

```text
[[3353  786]
 [ 409 1086]]
```

This analysis helps identify the trade-off between:

* False positives
* False negatives
* Precision
* Recall

For a churn prediction problem, false negatives are particularly important because they represent customers who were predicted as non-churners but actually churned.

---

## 🧪 Final Test Set Evaluation

The test set was kept separate from model development and was used only for the final evaluation.

### Final Model

```text
Model     : Gradient Boosting
Threshold : 0.34
```

### Final Test Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **77.50%** |
| Precision | **55.88%** |
| Recall    | **72.46%** |
| F1 Score  | **63.10%** |
| ROC-AUC   | **84.68%** |

### Final Confusion Matrix

```text
[[821 214]
 [103 271]]
```

### Classification Report

```text
              precision    recall  f1-score   support

No Churn          0.89      0.79      0.84      1035
Churn             0.56      0.72      0.63       374

accuracy                              0.78      1409
macro avg         0.72      0.76      0.73      1409
weighted avg      0.80      0.78      0.78      1409
```

---

## 📈 Model Evaluation Visualizations

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Precision-Recall Curve

![Precision Recall Curve](images/precision_recall_curve.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

---

## 🧠 SHAP Explainability

SHAP was used to make the final Gradient Boosting model more interpretable.

Instead of treating the model as a black box, SHAP helps answer:

> **Which features are influencing the model's churn predictions, and in which direction?**

### SHAP Summary

![SHAP Summary](images/shap_summary.png)

The SHAP summary provides a global view of feature importance and shows how different feature values influence the model's predictions.

### SHAP Waterfall

![SHAP Waterfall](images/shap_waterfall.png)

The waterfall plot explains the prediction of an individual customer by showing how each feature contributes to the final prediction.

---

## 💼 Business Interpretation

The model can be used to identify customers with a higher probability of churn.

A telecom company could use these predictions to:

* Identify high-risk customers
* Prioritize retention campaigns
* Offer personalized discounts
* Provide targeted customer support
* Investigate reasons for potential churn
* Allocate retention resources more efficiently

The threshold was deliberately reduced from `0.50` to `0.34` because identifying more potential churners was considered important for the business problem.

---

## 🔐 Data Leakage Prevention

The project follows a strict train/test methodology.

The test set was kept untouched during:

* Model comparison
* Cross-validation
* Hyperparameter tuning
* Threshold optimization

The test set was used only for the final evaluation.

This helps provide a more realistic estimate of how the final model performs on unseen data.

---

## 🛠️ Technologies Used

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SHAP
SciPy
Jupyter Notebook
```

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* Train/test splitting
* Feature scaling
* Categorical encoding
* Pipelines
* Logistic Regression
* Decision Trees
* Random Forest
* SVM
* Gradient Boosting
* Cross-validation
* ROC-AUC
* Precision
* Recall
* F1 Score
* Hyperparameter tuning
* Classification threshold optimization
* Confusion matrix analysis
* Error analysis
* Feature importance
* SHAP explainability
* Business interpretation
* Data leakage prevention

---

## 🚀 Future Improvements

Possible future improvements include:

* More extensive feature engineering
* Probability calibration
* Cost-sensitive learning
* Class-weight optimization
* Additional boosting algorithms
* Model monitoring
* Deployment as an API or web application
* Automated retraining pipeline

---

## 👨‍💻 Author

**Muhammad Tayseer**

Computer Engineering Student | Machine Learning & AI Engineering

---

## ⭐ Conclusion

This project demonstrates an end-to-end approach to solving a real-world customer churn prediction problem.

Rather than relying only on accuracy, the project evaluates multiple models, uses cross-validation and hyperparameter tuning, optimizes the classification threshold, performs error analysis, evaluates the final model on an untouched test set, and applies SHAP to explain model predictions.

**Final model: Gradient Boosting**

**Final ROC-AUC: 0.8468**

**Final Recall: 72.46%**

**Final F1 Score: 63.10%**
