## Implementation of Classification Model for Red Wine Quality Dataset
![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue)
![Docker](https://img.shields.io/badge/Docker%20Engine-19.03.12-blue)
![build](https://img.shields.io/badge/Build-Passing-brightgreen)
[![GHA](https://img.shields.io/github/v/tag/iterative/setup-cml?label=GitHub%20Actions&logo=GitHub)](https://github.com/iterative/setup-cml)

This repository contains the solution for classifying the quality of Wine on Kaggle dataset: . The selected models for classifying the quality of wine as "Good" or "Bad", includes Logistic Regression, Random Forest, Decison Tree.

The implementation can be run locally by building the Dockerfile. Continious Integration/Continious Deployments is performed by using the CML open source tool [Link to repository](https://github.com/iterative/cml) 

## 0. Problem Statement
Builiding of ML pipeline for Classifying the quality of red wine as "Good" or "Bad". The Problem was proposed in Kaggle challenge [Link to Dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

## 1. Description of Dataset
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

## 2. Preprocessing Steps Included
- Transformation of Target Variable(Quality) from the range 0-10 to 0(Bad) - 1(Good)
- Features Scaling Performed using StandardScaler  [Link to Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

## 3. Exploratory Data Analysis

