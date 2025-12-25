import os 
import sys 
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
    
def evaluate_models(X_train,y_train,X_test,y_test,models:dict,params):
    try:
        report = {}

        for model_name , model in models.items():
            param_grid = params.get(model_name,{})

            gs = GridSearchCV(estimator=model,param_grid = param_grid,cv = 3)
            gs.fit(X_train,y_train)
            best_model = gs.best_estimator_

            best_model.fit(X_train,y_train) # Train model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[model_name] = {
                "train_score" : train_model_score,
                "test_score" : test_model_score,
                "best_params": gs.best_params_
            }

        return report 

    except Exception as e:
        raise CustomException(e,sys)
