# Machine Learning Assignment 2: Classifier Performance Comparison Platform

## A. Problem Statement
This system simulates a highly robust **Customer Churn Prediction Engine** tailored to identify churn indicators. By leveraging continuous customer behavior signals across multiple dimensions, the platform accurately forecasts high-risk cohorts.

## B. Dataset Description
- **Total Rows**: 1000 instances
- **Features**: 12 predictive continuous variables generated synthetically via an informative structural manifold.
- **Target variable**: `target` (0: Stayed, 1: Churned)

## C. GitHub Repository Link
GitHub Repository: [https://github.com/<your-username>/<your-repo-name>](https://github.com/<your-username>/<your-repo-name>)

## D. Models Used & Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|:-----------------------|:---------|:-------|:----------|:-------|:---------|:-------|
| Logistic Regression    | 0.8150   | 0.8876 | 0.8247    | 0.8000 | 0.8122   | 0.6305 |
| Decision Tree          | 0.8800   | 0.8797 | 0.8713    | 0.8900 | 0.8806   | 0.7600 |
| KNN                    | 0.8700   | 0.9405 | 0.8571    | 0.8900 | 0.8732   | 0.7407 |
| Naive Bayes            | 0.8250   | 0.8914 | 0.8351    | 0.8100 | 0.8223   | 0.6506 |
| Support Vector Machine | 0.9250   | 0.9701 | 0.9057    | 0.9500 | 0.9273   | 0.8510 |

### Observations about Model Performance
1. **Support Vector Machine**: The SVM outperforms all other models with an outstanding AUC score of **0.9701** and an Accuracy of **92.50%**. Its high MCC shows excellent resilience to false classifications.
2. **KNN**: Performed strongly, achieving a high AUC of **0.9405**, highlighting that localized class clustering holds high descriptive capabilities for this dataset.
3. **Decision Tree Classifier**: Shows good accuracy of **88.00%** and captures non-linear splits well, but acts slightly more prone to minor variances compared to the SVM.
4. **Naive Bayes**: Shows strong reliability (Accuracy: **82.50%**) given its underlying independence assumptions.
5. **Logistic Regression**: Serves as a solid baseline showing linear separability accuracy of **81.50%**.

**Overall Winner**: **Support Vector Machine** due to its superiority in MCC, Accuracy, and near-perfect AUC alignment.