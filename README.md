The neural network model for assessing the likelihood of clinical pregnancy occurrence in a specific in vitro fertilization (IVF) protocol:

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

Upload the model files to the notebook "Calibrated PRAI calculation":
Neural network model: neural_network_model.h5
Logistic regression model: calibrated_model.pkl

## Dependencies
Ensure you have the following Python libraries installed:

- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- keras.models
- joblib
- sklearn.preprocessing
- seaborn
- matplotlib.pyplot

