from azureml.core import Workspace
import os
import json
import joblib
from azureml.core.model import Model
import pandas as pd

def init():
    global model
    model_path = Model.get_model_path('diabetes_hyperdrive_final')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        data = data["data"]
        df = pd.DataFrame.from_dict(data)
        result = model.predict(df)

        # if data is a dataset use below
        
        #data = json.loads(data)
        #df = pd.DataFrame.from_dict(data, orient="index")
        #result = model.predict(df)

        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

