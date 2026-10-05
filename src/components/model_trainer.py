import os
import sys

import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class ModelTrainer:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

    # ==========================================
    # Evaluation Function
    # ==========================================

    def evaluate_model(self, true, predicted):

        mae = mean_absolute_error(
            true,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                true,
                predicted
            )
        )

        r2 = r2_score(
            true,
            predicted
        )

        return mae, rmse, r2

    # ==========================================
    # Baseline Models
    # ==========================================

    def get_models(self):

        models = {

            "Linear Regression":
                LinearRegression(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(),

            "K-Neighbors Regressor":
                KNeighborsRegressor(),

            "Decision Tree Regressor":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest Regressor":
                RandomForestRegressor(
                    random_state=42
                ),

            "XGBoost":
                XGBRegressor(
                    random_state=42
                ),

            "CatBoost":
                CatBoostRegressor(
                    verbose=False,
                    random_state=42
                ),

            "AdaBoost Regressor":
                AdaBoostRegressor(
                    random_state=42
                )
        }

        return models

    # ==========================================
    # Hyperparameter Spaces
    # ==========================================

    def get_tuning_parameters(self):

        tuning_parameters = {

            "XGBoost": {

                "n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "max_depth": [
                    3,
                    4,
                    5,
                    6,
                    8
                ],

                "learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.2
                ],

                "subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ],

                "colsample_bytree": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            },

            "Random Forest Regressor": {

                "n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "max_depth": [
                    None,
                    5,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "max_features": [
                    "sqrt",
                    "log2",
                    None
                ]
            },

            "CatBoost": {

                "iterations": [
                    100,
                    200,
                    300,
                    500
                ],

                "depth": [
                    4,
                    5,
                    6,
                    8,
                    10
                ],

                "learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.2
                ],

                "l2_leaf_reg": [
                    1,
                    3,
                    5,
                    7,
                    10
                ]
            }
        }

        return tuning_parameters

    # ==========================================
    # Model Training
    # ==========================================

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info(
                "Model training started"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = self.get_models()

            results = []

            # ==================================
            # Baseline Model Training
            # ==================================

            for model_name, model in models.items():

                logger.info(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                train_pred = model.predict(
                    X_train
                )

                test_pred = model.predict(
                    X_test
                )

                train_mae, train_rmse, train_r2 = (
                    self.evaluate_model(
                        y_train,
                        train_pred
                    )
                )

                test_mae, test_rmse, test_r2 = (
                    self.evaluate_model(
                        y_test,
                        test_pred
                    )
                )

                results.append({
                    "Model": model_name,
                    "Train MAE": train_mae,
                    "Train RMSE": train_rmse,
                    "Train R2": train_r2,
                    "Test MAE": test_mae,
                    "Test RMSE": test_rmse,
                    "Test R2": test_r2
                })

            # ==================================
            # Tuning
            # ==================================

            tuning_parameters = (
                self.get_tuning_parameters()
            )

            tuned_models = {

                "XGBoost": XGBRegressor(
                    random_state=42
                ),

                "Random Forest Regressor":
                    RandomForestRegressor(
                        random_state=42
                    ),

                "CatBoost":
                    CatBoostRegressor(
                        verbose=False,
                        random_state=42
                    )
            }

            for model_name, model in tuned_models.items():

                logger.info(
                    f"Tuning {model_name}"
                )

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=
                        tuning_parameters[model_name],
                    n_iter=20,
                    scoring="r2",
                    cv=5,
                    random_state=42,
                    n_jobs=-1
                )

                search.fit(
                    X_train,
                    y_train
                )

                best_model = search.best_estimator_

                train_pred = best_model.predict(
                    X_train
                )

                test_pred = best_model.predict(
                    X_test
                )

                train_mae, train_rmse, train_r2 = (
                    self.evaluate_model(
                        y_train,
                        train_pred
                    )
                )

                test_mae, test_rmse, test_r2 = (
                    self.evaluate_model(
                        y_test,
                        test_pred
                    )
                )

                results.append({
                    "Model": f"{model_name} Tuned",
                    "Train MAE": train_mae,
                    "Train RMSE": train_rmse,
                    "Train R2": train_r2,
                    "Test MAE": test_mae,
                    "Test RMSE": test_rmse,
                    "Test R2": test_r2
                })

                logger.info(
                    f"{model_name} best parameters: "
                    f"{search.best_params_}"
                )

            # ==================================
            # Select Best Model
            # ==================================

            best_result = max(
                results,
                key=lambda x: x["Test R2"]
            )

            best_model_name = best_result["Model"]

            logger.info(
                f"Best model: {best_model_name}"
            )

            # Train / retrieve final best model
            if best_model_name.endswith("Tuned"):

                base_name = best_model_name.replace(
                    " Tuned",
                    ""
                )

                final_model = tuned_models[
                    base_name
                ]

                final_search = RandomizedSearchCV(
                    estimator=final_model,
                    param_distributions=
                        tuning_parameters[base_name],
                    n_iter=20,
                    scoring="r2",
                    cv=5,
                    random_state=42,
                    n_jobs=-1
                )

                final_search.fit(
                    X_train,
                    y_train
                )

                final_model = (
                    final_search.best_estimator_
                )

            else:

                final_model = models[
                    best_model_name
                ]

                final_model.fit(
                    X_train,
                    y_train
                )

            # ==================================
            # Save Best Model
            # ==================================

            save_object(
                file_path=self.model_path,
                obj=final_model
            )

            logger.info(
                "Best model saved successfully"
            )

            return results, best_model_name

        except Exception as e:

            logger.error(
                "Error occurred during model training"
            )

            raise CustomException(
                e,
                sys
            )