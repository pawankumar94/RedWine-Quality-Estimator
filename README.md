## Implementation of Classification Model for Red Wine Quality Dataset
![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue)
![Docker](https://img.shields.io/badge/Docker%20Engine-19.03.12-blue)
![build](https://img.shields.io/badge/Build-Passing-brightgreen)
[![GHA](https://img.shields.io/github/v/tag/iterative/setup-cml?label=GitHub%20Actions&logo=GitHub)](https://github.com/iterative/setup-cml)

This repository contains the solution for classifying the quality of Wine on Kaggle dataset: . The selected models for classifying the quality of wine as "Good" or "Bad", includes Logistic Regression, Random Forest, Decison Tree.

The implementation can be run locally by building the Dockerfile. Continious Integration/Continious Deployments is performed by using the CML open source tool [Link to repository](https://github.com/iterative/cml) 


## 0. Machine Learning Pipeline 
![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/855ea5dfb389f62fa9a25ef55dc4813d8d0ada61/graphics/MLPipeline.png)

## 1. Problem Statement
Builiding of ML pipeline for Classifying the quality of red wine as "Good" or "Bad". The Problem was proposed in Kaggle challenge [Link to Dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

## 2. Description of Dataset
- Columns:
    - Fixed Acidity: describes the fixed acidity of the wine
    - Volatile Acidity: describes the the amount of acetic acid in the wine
    - Citric Acid: describes the amount of citric acid in the wine. 
    - Residual Sugar: describes the amount of remaining sugar left after fermentation
    - Chlorides: describes the amount of salt present in wine
    - Free Sulphur Dioxide: describes the free form of SO2 present in Wine.
    - Total Sulphur Dioxide: describes the total amount of SO2 present in Wine
    - Density: describes the density of Wine
    - pH: describes the level of acidic or basix of the wine 
    - sulphates: describes the amount additive sulphate present in wine
    - alchol: describes the alchol percentage present in wine
    - quality: describes the quality of Wine (*Ranges from 0 -10, Wine Quality above 7 are considered as good.*)

## 3. Preprocessing Steps Included
- Transformation of Target Variable(Quality) from the range 0-10 to 0(Bad) - 1(Good)
- Features Scaling Performed using StandardScaler  [Link to Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

## 4. Exploratory Data Analysis
- As mentioned above the target variable(quality) ranges in a range between 0 - 10. We first transformed the problem into a binary classification problem, by moviing   all the instances with quality < 7 as "Bad" and quality >7 as "Good". At first glance on the below shown figures a high class imbalance problem in the dataset.

![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4078627b9531b5f6012d170a929973b94c7e9905/graphics/Ditribution.png)

![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4078627b9531b5f6012d170a929973b94c7e9905/graphics/before-oversample.png)

- The following Image shows the correlation matrix of the dataset. At first glance from the  below image, we can observe that  free sulphar dioxide and total sulphur dioxide have the same correlation between each other.   

![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4078627b9531b5f6012d170a929973b94c7e9905/graphics/correlation.png)

- The following Image shows us positive and negative correlation of the target variable with the features.
 
 ![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4078627b9531b5f6012d170a929973b94c7e9905/graphics/Correlation-quality.png)

## 5. Training and Monitoring Performance.

The models selected for solving the problem in question were:
1. Logistic Regression
2. Decison Tree
3. Random Forest

The models were trained on 70% of the dataset and we kept remaining 30% of dataset for testing. Due to the problem of class imbalance present in the dataset as shown in previous section. We first began by balancing out the classes in the training splitted dataset by using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) library oversampling technique which works on nearest neighbor algorithm , the chosen value of K for nearest neighbor was kept as **4**.

![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/b5d8cf8e5e27ec9b308aba84687dcc1c619d8c53/graphics/After-oversample.png)

The performance of the models were monitored by observing the results from following metrics:
- Training and Testing score 
- Confusion Matrix between predictions and actual values
- Classification Report of Each model

### 5.1 Logisitc Regression Results

Classification Report:
``` 
   precision    recall  f1-score   support

           0       0.98      0.77      0.86       413
           1       0.38      0.88      0.53        67

    accuracy                           0.79       480
   macro avg       0.68      0.83      0.70       480
weighted avg       0.89      0.79      0.82       480
```
![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4c29fe5fc01c813eaab69d1fd4df8d2d91ff9c32/Train-Test-Results/Logistic_Regression_confusion_matrix.png)

### 5.2 Decision Tree Results

Classification Report:
``` 
    precision    recall  f1-score   support

           0       0.95      0.88      0.91       413
           1       0.48      0.70      0.57        67

    accuracy                           0.85       480
   macro avg       0.71      0.79      0.74       480
weighted avg       0.88      0.85      0.86       480
```
![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4c29fe5fc01c813eaab69d1fd4df8d2d91ff9c32/Train-Test-Results/Decison_Tree_confusion_matrix.png)

### 5.3 Random Forest Results

Classification Report:
``` 
   precision    recall  f1-score   support

           0       0.96      0.91      0.93       413
           1       0.58      0.76      0.66        67

    accuracy                           0.89       480
   macro avg       0.77      0.84      0.80       480
weighted avg       0.91      0.89      0.90       480
```
![alt text](https://github.com/pawankumar94/RedWine-Quality-Estimator/blob/4c29fe5fc01c813eaab69d1fd4df8d2d91ff9c32/Train-Test-Results/Random_Forest_confusion_matrix.png)

### 5.4 Train and Test Score for all Models

|      Model Name      | Training Score | Testing Score |
|:--------------------:|:--------------:|:-------------:|
| Logistic  Regression |     79.1 %     |     78.5%     |
|    Decision  Tree    |     100.0%     |     85.2%     |
|    Random  Forest    |      99.8%     |     89.0%     |
