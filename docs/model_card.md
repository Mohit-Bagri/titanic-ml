# Titanic Survival Model Card

## Model
Logistic Regression with preprocessing pipeline

## Features
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Training
- Dataset: Kaggle Titanic
- CV: 5-fold Stratified
- Metric: ROC-AUC

## Performance
- CV ROC-AUC: ~0.84
- Test ROC-AUC: Stored in metadata

## Interpretability
SHAP feature importance saved in artifacts.

## Limitations
- Assumes clean inputs
- Trained on historical data
