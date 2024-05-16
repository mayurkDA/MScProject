# Fake News Detection: Content and Context Analysis

## Overview

This repository contains the code and data for the master's dissertation titled "Machine Learning and Mathematics for Fake News Detection" by Mayur Kripalani. The project focuses on developing a robust fake news detection system using a dual-analytical framework that combines Content and Context Analysis.

## Repository Structure

- **Codebase**: Contains all Python scripts used for data preprocessing, feature engineering, model training, and evaluation.
- **Data**: Includes datasets used for the project.
- **Reports**: PDF of the dissertation and additional documentation.

## Datasets

### TruthSeeker2023
- **Source**: Canadian Institute for Cybersecurity
- **Description**: Contains news articles from 2009 to 2022 and public tweets reacting to these articles. The dataset includes labeled tweets indicating whether the news is true or fake.

### LIAR Dataset
- **Source**: University of California, Santa Barbara
- **Description**: Contains textual and semantic attributes of news statements, including truth labels.

## Methodology

### Content Analysis

#### Data Preparation
1. **Import Datasets**: Load CSV files into Pandas DataFrames.
2. **Merge Datasets**: Combine datasets on unique identifiers and binary numerical targets.
3. **Data Cleaning**: Handle missing values using mean imputation for numerical columns and mode imputation for categorical columns.

#### Text Preprocessing
- **Tokenization and Lemmatization**: Use spaCy for tokenizing and lemmatizing text to convert it into a machine-readable format.

#### Feature Engineering
- **BERT Embeddings**: Use the BERT model to generate high-dimensional feature vectors representing the syntactic and semantic essence of news articles.

#### Deep Learning Model
- **Model Architecture**: Utilize a Bi-directional LSTM network with dropout layers for regularization.
- **Loss Function**: Binary cross-entropy to measure the discrepancy between actual and predicted labels.
- **Hyperparameter Optimization**: Use Optuna for optimizing learning rate, number of layers, and dropout rate.

### Context Analysis

#### Bot Behavior Analysis
- **Feature Selection**: Choose relevant features like followers count, friends count, and tweet frequency.
- **Random Forest Regressor**: Train a model to predict 'BotScore' for each sample.
- **Evaluation Metrics**: Use MSE and RMSE to evaluate model performance.

#### Sentiment Analysis
- **BERT for Sentiment Analysis**: Leverage BERT's capabilities for a more granular and accurate sentiment analysis of tweets.

#### Majority Target Prediction
- **XGBoost Classifier**: Use XGBoost to predict 'Majority Target', indicating the veracity of news articles.

#### Final Prediction Model
- **Random Forest Classifier**: Integrate features like predicted BotScore, sentiment score, and predicted majority target for the final classification of news articles.

### Content Analysis of LIAR Dataset
- **Data Preprocessing**: Handle missing values and scale numerical features.
- **Feature Selection**: Include textual, numerical, and categorical features.
- **Model Architecture**: Hybrid neural network with densely connected layers, Bi-LSTM layers, and dropout layers.
- **Optimization and Evaluation**: Use Keras Tuner for hyperparameter tuning and evaluate the model using K-Fold cross-validation.

## Results and Analysis

### TruthSeeker2023 Dataset
- **Text Preprocessing and Feature Extraction**: Achieved high accuracy with BERT embeddings and Bi-LSTM model.
- **Bot Behavior Analysis**: Effective in predicting bot scores with Random Forest Regressor.
- **Sentiment and Majority Target Prediction**: Enhanced prediction using BERT for sentiment analysis and XGBoost for majority target.

### LIAR Dataset
- **Feature Engineering and Model Training**: Effective feature selection and preprocessing steps led to a robust model with high accuracy in truth label prediction.

## Conclusion

This project successfully integrates Content and Context Analysis to develop a comprehensive fake news detection system. The results demonstrate the effectiveness of using advanced NLP techniques like BERT and robust machine learning models like Random Forest and XGBoost.

## Acknowledgements

Special thanks to Professor Sergei K. Turitsyn, Dr. Pedro Freire, and Mrs. Nataliia Manuilovich for their invaluable guidance and support throughout this project.
"""
