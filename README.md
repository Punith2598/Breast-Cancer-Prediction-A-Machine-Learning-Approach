# Breast-Cancer-Prediction-A-Machine-Learning-Approach

## Overview
This project applies machine learning techniques to predict breast cancer diagnoses using the Wisconsin Breast Cancer dataset. The objective is to classify tumors as malignant or benign based on various features. This repository includes the complete workflow from data loading, preprocessing, model building, evaluation, to final predictions.

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
6. [Model Building and Evaluation](#model-building-and-evaluation)
7. [Model Tuning](#model-tuning)
8. [Predictions](#predictions)
9. [Conclusion](#conclusion)
10. [References](#references)

## Introduction
This project presents a detailed analysis and modeling process for breast cancer classification. The goal is to classify tumors as malignant (M) or benign (B) using machine learning algorithms. The process involves data exploration, preprocessing, visualization, model building, evaluation, and tuning to achieve optimal prediction performance.

## Data
The dataset used is the Wisconsin Breast Cancer Dataset, which includes features computed from breast cancer cell images.

Files included:
- `data.csv`: The dataset used for training and evaluation.
- `submission.csv`: The final predictions made by the best model.

## Installation
To run this project, you need to have Python installed along with the following libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Punith2598/Breast-Cancer-Prediction-A-Machine-Learning-Approach.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Breast-Cancer-Prediction-A-Machine-Learning-Approach
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook 'Breast Cancer Wisconsin (Diagnostic).ipynb'
   ```

## Data Exploration and Preprocessing
1. **Data Loading and Inspection**: The dataset contains 569 entries with 33 columns including 'diagnosis' and various feature measurements.
2. **Missing Value Analysis**: Identified and removed the 'Unnamed: 32' column which was completely empty.
3. **Feature Standardization**: All numeric columns were standardized using `StandardScaler`.
4. **Target Variable Encoding**: The 'diagnosis' column was encoded to binary values (M: 1, B: 0).

## Model Building and Evaluation
1. **Model Selection**: Evaluated five models - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machine (SVM).
2. **Training Process**: Models were trained using an 80-20 train-validation split.
3. **Evaluation Metrics**: Models were evaluated using accuracy, precision, recall, and F1 score. Logistic Regression and SVM achieved the highest F1 scores of 0.9647.

## Model Tuning
1. **Hyperparameter Tuning**: For Logistic Regression, a grid search with cross-validation was performed to find the optimal parameters (C=1, penalty='l1', solver='liblinear').
2. **Performance Improvement**: The tuned Logistic Regression model achieved an improved F1 score of 0.9655.

## Predictions
- Predictions were made on the test set.
- The submission file `submission.csv` was prepared for potential use in a competition or further analysis.

## Conclusion
This machine learning model demonstrates high accuracy in predicting breast cancer diagnoses, potentially aiding in early detection and treatment planning. Future work could involve further optimization, integration into clinical workflows, and continuous improvement with new data.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

This README covers all the essential aspects of your project and provides clear instructions and explanations for users to follow and understand your work.
