# The neural network model for assessing the likelihood of clinical pregnancy occurrence in a specific in vitro fertilization (IVF) protocol:

The developed deep neural network serves as a unique tool for IVF clinics, aiming to enhance the efficiency of internal quality control and the determination of its target indicators. This model, based on recurrent neural networks, exhibits high accuracy in predicting the occurrence or absence of clinical pregnancy in a specific IVF protocol (AUC 0.68-0.86; Test accuracy 0.78, F1 Score 0.71).

Dataset for Training: new_df_with_KPI.xlsx
This repository includes the following components:

Data Engineering:

Code for data preprocessing.
Data Split for Neural Network:

Code for splitting the dataset into training, validation, and test sets.
Model Development:

Draft version of the model development stages and training.
Model from Nature:

Building a model based on published works with a complex structure.
Model Evaluation Metrics:

Evaluation of the model's quality using various metrics.

Instructions for Google Colab Usage:
The provided code is designed to be run in Google Colab. To use the model for predictions, follow these steps:

# KPI Calculation and Pregnancy Outcome Prediction

This repository includes a Python script (`kpi_and_prediction_script.py`) for calculating KPI (Key Performance Indicators) scores based on certain criteria and predicting pregnancy outcomes using a pre-trained neural network model.

## Neural Network Model Loading

The script begins by loading a pre-trained neural network model (`neural_network_model.h5`) using Keras and a calibrated logistic regression model (`calibrated_model.pkl`) using joblib.

## KPI Calculation and Dataset Preparation

1. Data is loaded from an Excel file (`*.xlsx`) into a Pandas DataFrame (`new_df`).
2. Missing values are replaced with 0, and numeric conversion is applied to specific columns.
3. Several new features are calculated based on existing columns.
4. A KPI score is calculated for each row using a defined function (`calculate_kpi_score()`).
5. The KPI scores are added to the DataFrame as a new column (`KPIScore`).
6. The DataFrame is saved to a new Excel file (`new_df_with_KPI.xlsx`).

## Feature Selection and Data Normalization

1. Features for prediction are selected.
2. Standard scaling is applied to normalize the data.
3. The data is reshaped to fit the model input format.

## Model Predictions and Calibration

1. Neural network model predictions are obtained.
2. If the calibrated model is available, predictions are calibrated using logistic regression.
3. Predictions are saved to an Excel file (`вероятности.xlsx`).

## Decision Threshold and Result Analysis

1. A decision threshold of 0.48 is set.
2. Predictions are categorized as "да" or "нет" based on the threshold.
3. Results are saved to an Excel file (`предсказания.xlsx`).
4. The script prints the frequency of positive outcomes.

## Actual vs. Predicted Visualization

1. A bar chart is created to visually compare actual and predicted outcomes.

## Dependencies

Ensure you have the following Python libraries installed:

- Keras
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Seaborn
- Matplotlib

