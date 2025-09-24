Weather-Based Outfit Recommender
A simple and intelligent application that recommends outfits based on current weather conditions, temperature, humidity, and season. This tool uses a machine learning model trained on a dataset of weather scenarios and corresponding suitable outfits.

Features
Predicts outfit suggestions based on:
Temperature
Humidity
Season (Winter, Summer, Spring, Autumn, Monsoon)
Weather condition (Sunny, Rainy, Snowy, Windy, Cloudy, Foggy)
Provides precise recommendations for daily wear.
User-friendly web interface with a responsive design.
Offline fallback: Provides outfit suggestions even if the server is unavailable.

Technology Stack
Backend: Python, Flask
Frontend: HTML, CSS, JavaScript
Machine Learning: Scikit-learn (RandomForestClassifier)
Data Processing: Pandas, LabelEncoder
Model Persistence: Joblib
CORS: Flask-CORS for cross-origin requests
