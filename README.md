# ML Data Challenge

This repository contains solutions to two supervised learning problems as part of the Machine Learning course at Bocconi University. The tasks involve real-world datasets and require both predictive modeling and cost-sensitive optimization.

## Contents

- `spam_detection/`: Classify emails as spam or non-spam
- `call_forecasting/`: Predict the number of outgoing calls for telecom customers

## Problem 1: Spam Detection

**Goal**: Build a classifier to detect spam emails and minimize a cost-sensitive loss.

- **Data**: 4601 emails  
  - 3101 labeled (spam = 2, non-spam = 1)
  - 1500 unlabeled (for prediction)

- **Features**:
  - 48 word frequency percentages (e.g., "business", "internet")
  - 6 character frequency features (e.g., `charDollar`, `charExclamation`)
  - 3 capitalization pattern metrics (`capitalAve`, `capitalLong`, `capitalTotal`)

- **Cost-sensitive classification**:
  - False positive (non-spam → spam): cost = 5
  - False negative (spam → non-spam): cost = 1

- **Output**: `spam_predictions.csv` with predicted labels for the 1500 unlabeled emails

## Problem 2: Outgoing Calls Forecasting

**Goal**: Predict the total number of outgoing calls in month 9 for 5300 telecom customers based on previous behavior.

- **Data**: 15,300 customer records  
  - 10,000 labeled with number of calls (`y`)
  - 5,300 unlabeled (for prediction)

- **Features** (88 total):
  - Customer profile: age, gender, area, tariff plan, payment method, etc.
  - Contract data: activation channel, added services
  - Monthly behavior data for months 1–8:
    - Outgoing calls (peak/off-peak): count, duration, value
    - Incoming calls
    - SMS count
    - Customer service contact count

- **Output**: `call_predictions.csv` with predicted number of calls for the 5300 held-out customers

## Repository Structure

```
ML-Data-Challenge/
├── spam_detection/
│   ├── spam_model.ipynb         # Model training and cost-sensitive evaluation
│   ├── predictions.csv          # Final predicted labels for 1500 unlabeled emails
│   └── ...
│
├── call_forecasting/
│   ├── call_model.ipynb         # Regression model for call prediction
│   ├── predictions.csv          # Final predictions for 5300 customers
│   └── ...
│
└── README.md
```

## Tools & Techniques

- Logistic Regression, XGBoost, Random Forest
- Feature scaling, one-hot encoding
- Cross-validation
- Threshold tuning for cost-sensitive classification
- Evaluation metrics: accuracy, RMSE, misclassification cost

