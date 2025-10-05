import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_clean_data

MODEL_PATH = "./heart_model.pkl"

def train_model(file_path="US_Heart_Patients.csv"):
    df, _ = load_and_clean_data(file_path)
    X = df.drop("heart disease", axis=1)
    y = df["heart disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved. Accuracy: {acc:.2f}")
    return model

def load_model():
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    # Train the model and save it as heart_model.pkl
    train_model("US_Heart_Patients.csv")