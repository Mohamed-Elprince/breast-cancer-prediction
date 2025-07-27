Breast Cancer Prediction using Logistic Regression, Random Forest, and KNN
This project explores breast cancer prediction using multiple machine learning models on the Breast Cancer Wisconsin dataset
The goal is to identify high-risk cancer cases based on patients' medical and diagnostic data.

Files in this Repo
breast_cancer_prediction.ipynb – Main Jupyter notebook with code, EDA, modeling, and evaluation.

README.md – This file.

Tools & Libraries
- Python

- pandas, numpy

- scikit-learn – for modeling, preprocessing, and evaluation

Project Steps
1. Data Loading 
- Loaded the dataset using sklearn.datasets.load_breast_cancer()

2. Logistic Regression Model
- Used LogisticRegression(class_weight='balanced') to address class imbalance

- Split data (80/20) into training and test sets

- Accuracy improved by ~2% with class weights

- Evaluated using accuracy, confusion matrix, precision, recall, and F1 score

3. Random Forest Model
- Trained a RandomForestClassifier as an alternative approach

- Compared performance to logistic regression using:

- Accuracy

- Confusion Matrix

- Cross-Validation scores

4. K-Nearest Neighbors (KNN)
- Trained a KNeighborsClassifier with k=5

- Achieved 100% recall for cancer cases, which is critical in medical diagnosis

- Slight decrease in overall accuracy, but improved cancer detection

5. Cross-Validation
- Ran 5-fold cross-validation for all models

- Compared average accuracy to evaluate generalization

- Helped confirm model stability across different data subsets

Summary
This project demonstrates how different machine learning algorithms perform on a medical dataset, highlighting the importance of handling class imbalance and choosing metrics beyond accuracy when working on life-critical problems.