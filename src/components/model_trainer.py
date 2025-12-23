import os
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
#from src.components.data_ingestion import DataIngestion
#from src.components.data_transformation import DataTransformation
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours Classifiers":KNeighborsRegressor(),
                "XGBClassifier" : XGBRFRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose = False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
    
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion
#     obj = DataIngestion()
#     train_data,test_data = obj.initiate_data_ingestion()
    
#     from src.components.data_transformation import DataTransformation
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_arr,test_arr))