# Heart Disease Prediction using Machine Learning


This GitHub repository contains a comprehensive machine learning model for predicting heart disease based on various patient attributes. The model is built using Python and follows a structured workflow. Below, you'll find detailed information on each aspect of this project.

## Table of Contents

- [Introduction](#introduction)
- [Workflow](#workflow)
- [Getting Started](#getting-started)
- [Data Collection and Processing](#data-collection-and-processing)
- [Data Visualization](#data-visualization)
- [Splitting the Features and Target](#splitting-the-features-and-target)
- [Data Standardization](#data-standardization)
- [Splitting the Data into Training and Test Sets](#splitting-the-data-into-training-and-test-sets)
- [Model Training](#model-training)
  - [Logistic Regression](#1-logistic-regression)
  - [Naive Bayes Classifier](#2-naive-bayes-classifier)
  - [K-Nearest Neighbor (KNN)](#3-k-nearest-neighbor-knn)
  - [Decision Tree Classifier](#4-decision-tree-classifier)
  - [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
- [Multi-model Training](#multi-model-training)
- [Model Improvement](#model-improvement)
- [Model Evaluation](#model-evaluation)
- [Building a Prediction System](#building-a-prediction-system)
- [Notations](#notations)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease is a major health concern worldwide. Early detection and prediction of heart disease can significantly improve patient outcomes. This project aims to develop a machine learning model capable of predicting heart disease based on a set of patient attributes.

## Workflow

The project follows a structured workflow, including data collection, preprocessing, model training, evaluation, and deployment. Here is an overview of the workflow:

### 1. Data Collection and Processing

- Data is collected from a CSV file (`heart.csv`) containing patient information.
- Basic data exploration and inspection are performed to understand the dataset.

### 2. Data Visualization

- Data is visualized using popular Python libraries such as Seaborn and Matplotlib to gain insights into feature distributions and relationships.
- Pair plots and heatmaps are used for visualization.

### 3. Splitting the Features and Target

- Features (attributes) and the target (heart disease status) are separated to prepare the data for model training.

### 4. Data Standardization

- Standardization is applied to the feature data to ensure that all features have the same scale.

### 5. Splitting the Data into Training and Test Sets

- The dataset is split into training and test sets to evaluate the model's performance.
- The `stratify` parameter is used to ensure a balanced distribution of target classes in both sets.

### 6. Model Training

#### 1. Logistic Regression

- Logistic Regression is applied to the training data.
- Model accuracy is evaluated.

#### 2. Naive Bayes Classifier

- A Gaussian Naive Bayes Classifier is trained.
- Model accuracy is evaluated.

#### 3. K-Nearest Neighbor (KNN)

- A K-Nearest Neighbor Classifier with `n_neighbors=7` is trained.
- Model accuracy is evaluated.

#### 4. Decision Tree Classifier

- A Decision Tree Classifier is created.
- Model accuracy is evaluated.
- Possible overfitting is identified.

#### 5. Support Vector Machine (SVM)

- A Support Vector Machine with a linear kernel is trained.
- Model accuracy is evaluated.

### 7. Multi-model Training

- Additional machine learning models, including Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost, are trained and evaluated for performance.

### 8. Model Improvement

- A Voting Classifier is implemented to combine the predictions from multiple models for improved accuracy and precision.

### 9. Model Evaluation

- The model's performance is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.
- Metrics are reported for both the training and test datasets.

### 10. Building a Prediction System

- A simple prediction system is built, allowing users to input patient attributes and receive predictions about heart disease.

### 11. Notations

- Clear notations are provided to interpret model predictions.
- [0]: Indicates that the patient has a healthy heart.
- [1]: Indicates that the patient has a heart disease.

### 12. Saving the Model

- The trained model is saved to a file for later use.
- Both `pickle` and `joblib` libraries are used for model persistence.

## Getting Started

To use this project:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/Heart_Disease_Prediction_model_using_Machine_Learning.git
   ```

2. Install the required Python packages:

   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost joblib
   ```

3. Open the Jupyter Notebook (`heart_disease_prediction.ipynb`) to execute the code and explore the project.

4. Utilize the saved model (`trained_model.pkl` or `heart_model.sav`) for making heart disease predictions with new data.

## Usage

To make predictions using the saved model:

```python
import pickle

# Load the saved model
loaded_model = pickle.load(open("trained_model.pkl", "rb"))

# Example input data
input_data = (58, 0, 3, 150, 283, 1, 0, 162, 0, 1, 2, 0, 2)

# Convert input data to a numpy array and reshape it
input_data_array = np.asarray(input_data)
input_data_reshaped = input_data_array.reshape(1, -1)

# Predict the result
prediction = loaded_model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("Patient has a healthy heart")
else:
    print("Patient has a heart disease")
```

## Contributing

Contributions to this project are welcome. Feel free to submit issues, suggestions, or pull requests to improve the model or the documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Note: This README provides a detailed overview of the Heart Disease Prediction project. For in-depth code and details, refer to the Jupyter Notebook in the repository.*
