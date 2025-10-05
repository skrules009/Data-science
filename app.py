from flask import Flask, request, jsonify
import pandas as pd
from model import load_model
from preprocess import load_and_clean_data

app = Flask(__name__)
model = load_model()

@app.route("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data]) 

        X, _ = load_and_clean_data("US_Heart_Patients.csv")
        df = df.reindex(columns=X.drop("heart disease", axis=1).columns, fill_value=0)

        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)