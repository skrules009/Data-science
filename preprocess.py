import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if "AgeGroup" not in df.columns:
        df["AgeGroup"] = pd.cut(
            df["age"], 
            bins=[29, 40, 50, 60, 70, 80], 
            labels=["30-40", "41-50", "51-60", "61-70", "71-80"]
        )

    # Convert AgeGroup to numeric midpoint
    def age_group_to_midpoint(age_group):
        start, end = map(int, age_group.split('-'))
        return (start + end) / 2

    df["AgeGroup"] = df["AgeGroup"].apply(age_group_to_midpoint)

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col != "AgeGroup":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders