# Student Performance Indicator

A Machine Learning regression project that predicts a student's **Mathematics Score** based on demographic information, parental education, lunch type, test preparation, reading score, and writing score.

## Project Overview

The project follows a modular and reusable Machine Learning pipeline:

- Data Ingestion
- Data Transformation
- Model Training
- Model Evaluation
- Model Serialization
- Prediction Pipeline
- Flask Web Application

The project is organized into separate components so that data processing, model training, and prediction can be maintained and reused independently.

## Dataset

The dataset is stored at:

```text
notebook/data/stud.csv
```

It contains student demographic information and academic scores.

### Features

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

### Target

- `math_score`

## Machine Learning Models

The following regression models are evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

Hyperparameter tuning is performed for selected ensemble models using `RandomizedSearchCV`.

After model comparison and tuning, **Ridge Regression achieved the best test performance** among the evaluated models in this project.

## Project Structure

```text
D:\Projects
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
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── notebook/
│   ├── data/
│   │   └── stud.csv
│   ├── 1. EDA STUDENT PERFORMANCE .ipynb
│   └── 2. MODEL TRAINING .ipynb
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── test.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Project Workflow

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Train / Test Split
     │
     ▼
Data Transformation
     │
     ├── Numerical Imputation
     ├── Standard Scaling
     ├── Categorical Imputation
     └── One-Hot Encoding
     │
     ▼
Model Training
     │
     ├── Baseline Models
     └── Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ▼
Best Model
     │
     ▼
Model Serialization
     │
     ▼
Prediction Pipeline
     │
     ▼
Flask Web Application
```

## Model Artifacts

The trained production model and preprocessing object are stored in the `artifacts` directory:

- `artifacts/model.pkl` — trained final model
- `artifacts/preprocessor.pkl` — fitted preprocessing pipeline
- `artifacts/train.csv` — training dataset generated during data ingestion
- `artifacts/test.csv` — test dataset generated during data ingestion

## Prediction

The prediction pipeline accepts student information and returns the predicted Mathematics Score.

Example input fields:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

## Flask Web Application

The project includes a Flask web application that provides a simple interface for making predictions.

The application contains:

- `templates/index.html` — landing page
- `templates/home.html` — prediction form
- `app.py` — Flask application and prediction routes

The prediction flow is:

```text
User Input
    ↓
Flask Application
    ↓
CustomData
    ↓
Preprocessor
    ↓
Trained Model
    ↓
Predicted Math Score
```

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run Training Pipeline

From the project root:

```bash
python src/pipeline/train_pipeline.py
```

This performs:

1. Data ingestion
2. Train-test split
3. Data transformation
4. Model training
5. Model evaluation
6. Final model serialization

## Run Prediction Test

```bash
python test.py
```

This tests the prediction pipeline using custom student input.

## Run Flask Application

The Flask application can be started with:

```bash
python app.py
```

For the project's production-style local server setup, the application can also be served using Waitress.

Example:

```bash
python run.py
```

Then open the local application URL shown by the server.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Flask
- Waitress
- Matplotlib
- Seaborn
- Jupyter Notebook

## Key Concepts Demonstrated

- Modular Machine Learning project structure
- Train-Test Split
- Data Preprocessing
- Missing Value Handling
- One-Hot Encoding
- Feature Scaling
- Regression Algorithms
- Hyperparameter Tuning
- Model Evaluation
- Model Serialization
- Reusable Prediction Pipeline
- Flask Deployment

## Author

**Aman Soni**
