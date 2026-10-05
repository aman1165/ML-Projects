# Student Performance Indicator

A Machine Learning regression project that predicts a student's mathematics score based on demographic information, parental education, lunch type, test preparation, reading score, and writing score.

## Project Overview

The project follows a modular and reusable Machine Learning pipeline:

- Data Ingestion
- Data Transformation
- Model Training
- Model Evaluation
- Model Serialization
- Prediction Pipeline
- Flask Web Application

## Dataset

The dataset contains student information and academic scores.

### Features

- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score

### Target

- math_score

## Machine Learning Models

The following regression models are evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost
- CatBoost
- AdaBoost Regressor

Hyperparameter tuning is performed for selected ensemble models using `RandomizedSearchCV`.

## Project Structure

```text
ML_Project_Template/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
│
├── artifacts/
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── notebooks/
│
├── app.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore