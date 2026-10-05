import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


class DataIngestion:

    def __init__(self):

        self.data_path = os.path.join(
            "notebook",
            "data",
            "stud.csv"
        )

        self.train_data_path = os.path.join(
            "artifacts",
            "train.csv"
        )

        self.test_data_path = os.path.join(
            "artifacts",
            "test.csv"
        )

    def initiate_data_ingestion(self):

        logger.info("Data ingestion started")

        try:

            # Read raw dataset
            df = pd.read_csv(self.data_path)

            logger.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logger.info(
                "Train test split completed"
            )

            # Save train data
            train_set.to_csv(
                self.train_data_path,
                index=False,
                header=True
            )

            # Save test data
            test_set.to_csv(
                self.test_data_path,
                index=False,
                header=True
            )

            logger.info(
                "Train and test data saved successfully"
            )

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:

            logger.error(
                "Error occurred during data ingestion"
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = (
        obj.initiate_data_ingestion()
    )

    print("Data Ingestion Completed")

    print(f"Train Data: {train_data}")
    print(f"Test Data: {test_data}")