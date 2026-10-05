import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):

        try:

            # ==========================================
            # 1. Data Ingestion
            # ==========================================

            logger.info("Starting Data Ingestion")

            data_ingestion = DataIngestion()

            train_data_path, test_data_path = (
                data_ingestion.initiate_data_ingestion()
            )

            logger.info("Data Ingestion completed")


            # ==========================================
            # 2. Data Transformation
            # ==========================================

            logger.info("Starting Data Transformation")

            data_transformation = DataTransformation()

            train_arr, test_arr, _ = (
                data_transformation.initiate_data_transformation(
                    train_data_path,
                    test_data_path
                )
            )

            logger.info("Data Transformation completed")


            # ==========================================
            # 3. Model Training
            # ==========================================

            logger.info("Starting Model Training")

            model_trainer = ModelTrainer()

            results, best_model = (
                model_trainer.initiate_model_trainer(
                    train_arr,
                    test_arr
                )
            )

            logger.info(
                f"Model Training completed. "
                f"Best Model: {best_model}"
            )

            return results, best_model

        except Exception as e:

            logger.error(
                "Error occurred during training pipeline"
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    results, best_model = (
        pipeline.run_pipeline()
    )

    print("\nTraining Pipeline Completed!")
    print(f"Best Model: {best_model}")

    print("\nModel Results:")

    for result in results:
        print(result)