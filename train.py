import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("outfit.csv")  # same folder

# Encode categorical columns
le_season = LabelEncoder()
le_weather = LabelEncoder()
le_outfit = LabelEncoder()

data["season"] = le_season.fit_transform(data["season"])
data["weather"] = le_weather.fit_transform(data["weather"])
data["outfit"] = le_outfit.fit_transform(data["outfit"])

# Features and target
X = data[["temperature", "humidity", "season", "weather"]]
y = data["outfit"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save model + encoders in the same folder
joblib.dump(model, "outfit_model.pkl")
joblib.dump(le_season, "season_encoder.pkl")
joblib.dump(le_weather, "weather_encoder.pkl")
joblib.dump(le_outfit, "outfit_encoder.pkl")

print("Model trained and saved!")