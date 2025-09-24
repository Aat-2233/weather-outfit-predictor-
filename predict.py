
import sys
import joblib

# Load model + encoders from the same folder
model = joblib.load("outfit_model.pkl")
le_season = joblib.load("season_encoder.pkl")
le_weather = joblib.load("weather_encoder.pkl")
le_outfit = joblib.load("outfit_encoder.pkl")

# Command-line argument check
if len(sys.argv) != 5:
    print("Usage: python predict.py <temperature> <humidity> <season> <weather>")
    print("Example: python predict.py 12 75 Winter Rainy")
    sys.exit(1)

# Read and validate numeric input
try:
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
except ValueError:
    print("Temperature and humidity must be numbers!")
    sys.exit(1)

if not (0 <= temp <= 40):
    print("Temperature must be between 0 and 40°C.")
    sys.exit(1)

if not (0 <= humidity <= 100):
    print("Humidity must be between 0 and 100%.")
    sys.exit(1)

# Capitalize input to match encoder classes
season = sys.argv[3].capitalize()   # e.g., "Winter"
weather = sys.argv[4].capitalize()  # e.g., "Sunny"

# Encode season and weather
try:
    season_enc = le_season.transform([season])[0]
    weather_enc = le_weather.transform([weather])[0]
except ValueError:
    print("Unknown season or weather! Check your input.")
    print("Valid seasons:", list(le_season.classes_))
    print("Valid weathers:", list(le_weather.classes_))
    sys.exit(1)

# Predict outfit
prediction = model.predict([[temp, humidity, season_enc, weather_enc]])
outfit = le_outfit.inverse_transform(prediction)[0]

print("Recommended Outfit:", outfit.title())

