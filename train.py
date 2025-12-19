import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Load data
df = pd.read_csv("flight_data.csv")

df["is_delayed"] = (df["delay_minutes"] > 15).astype(int)

X = df.drop(["delay_minutes", "is_delayed"], axis=1)
y = df["is_delayed"]

numeric_features = ["dep_hour"]
categorical_features = ["weather", "route", "aircraft_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model.pkl")
print("✅ Model trained and saved as model.pkl")
