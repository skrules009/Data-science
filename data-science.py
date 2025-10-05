from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from math import sqrt
import pickle
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore")


data = pd.read_csv('US_Heart_Patients.csv',index_col=0)

data.head(10)
data.describe().T
summary = data.quantile([0, 0.25, 0.5, 0.75, 1.0]).T
summary.columns = ["min", "Q1", "median", "Q3", "max"]
summary

# Load the saved pipeline
with open("final_heart_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Input as JSON
    # Example input: {"age": 56, "sex": 1, "BP": 140, "cholestrol": 240, "AgeGroup": "Middle-aged"}
    
    # Convert to dataframe (1 row)
    import pandas as pd
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]
    
    return jsonify({"prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)