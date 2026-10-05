import os
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer_object(self):

        try:
            numerical_columns = [
                "reading_score",
                "writing_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            logger.error(
                "Error occurred while creating preprocessor"
            )
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and test data loaded successfully")

            # Target column
            target_column_name = "math_score"

            # Split input and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logger.info("Input and target features separated")

            # Create preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            # Fit on train, transform train and test
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            logger.info(
                "Preprocessing completed successfully"
            )

            # Save preprocessor
            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info(
                "Preprocessor saved successfully"
            )

            # Combine features + target
            train_arr = pd.concat(
                [
                    pd.DataFrame(input_feature_train_arr),
                    target_feature_train_df.reset_index(drop=True)
                ],
                axis=1
            ).values

            test_arr = pd.concat(
                [
                    pd.DataFrame(input_feature_test_arr),
                    target_feature_test_df.reset_index(drop=True)
                ],
                axis=1
            ).values

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path
            )

        except Exception as e:

            logger.error(
                "Error occurred during data transformation"
            )

            raise CustomException(e, sys)