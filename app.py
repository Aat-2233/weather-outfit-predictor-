from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # 👈 Import CORS
import joblib

# Load model and encoders
model = joblib.load("outfit_model.pkl")
le_season = joblib.load("season_encoder.pkl")
le_weather = joblib.load("weather_encoder.pkl")
le_outfit = joblib.load("outfit_encoder.pkl")

app = Flask(__name__)
CORS(app) # 👈 Enable CORS for all routes

@app.route("/")
def home():
    return render_template("indexx.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        temp = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
    except (TypeError, ValueError):
        return jsonify({"error": "Temperature and humidity must be numbers"}), 400

    season = data.get("season").capitalize()
    weather = data.get("weather").capitalize()

    # Encode
    try:
        season_enc = le_season.transform([season])[0]
        weather_enc = le_weather.transform([weather])[0]
    except ValueError:
        return jsonify({
            "error": "Unknown season or weather!",
            "valid_seasons": list(le_season.classes_),
            "valid_weathers": list(le_weather.classes_)
        }), 400

    # Predict
    prediction = model.predict([[temp, humidity, season_enc, weather_enc]])
    outfit = le_outfit.inverse_transform(prediction)[0]

    return jsonify({"outfit": outfit})

if __name__ == "__main__":
    app.run(debug=True)