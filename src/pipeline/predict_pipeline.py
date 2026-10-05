import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features):

        try:

            logger.info("Prediction started")

            # Load preprocessor
            preprocessor = load_object(
                self.preprocessor_path
            )

            # Load trained model
            model = load_object(
                self.model_path
            )

            # Transform new data
            data_scaled = preprocessor.transform(
                features
            )

            # Make prediction
            prediction = model.predict(
                data_scaled
            )

            logger.info("Prediction completed successfully")

            return prediction

        except Exception as e:

            logger.error(
                "Error occurred during prediction"
            )

            raise CustomException(
                e,
                sys
            )


class CustomData:

    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = (
            parental_level_of_education
        )
        self.lunch = lunch
        self.test_preparation_course = (
            test_preparation_course
        )
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "gender": [self.gender],

                "race_ethnicity": [
                    self.race_ethnicity
                ],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [self.lunch],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [
                    self.reading_score
                ],

                "writing_score": [
                    self.writing_score
                ]
            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:

            logger.error(
                "Error creating custom data DataFrame"
            )

            raise CustomException(
                e,
                sys
            )