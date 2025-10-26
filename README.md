
# Flask ED Visit Forecast API

This repository is part of of emergency medicine research which provides a Flask-based REST API server for forecasting Emergency Department (ED) visits. The API uses a machine learning model (Keras) and weather data from the Open-Meteo API to generate 28-day forecasts (14 days prior and 14 days ahead of current date), incorporating features such as temperature, humidity, holidays, and day-of-week effects.

Steps to create LSTM model:
https://www.kaggle.com/code/sufiansafaai/emergency-department-visit-prediction-using-lstm

## Features
- Fetches and processes weather data for Kuantan, Pahang
- Annotates dates with Malaysian national and Pahang state holidays
- Preprocesses input data and generates forecasts using a trained deep learning model
- REST endpoints:
	- `/` : Fetches weather data, annotates holidays, and returns formatted forecast results
	- `/forecast` : Accepts custom sequences for direct forecasting

## Project Structure
- `app.py` — Main Flask application and API logic
- `model/` — Trained Keras model and preprocessing artifacts. makesure for your files named as forecast_model.keras forecast_model_preprocessor.pkl and forecast_model_model_info.pkl
- `requirements.txt` — Python dependencies

## Setup & Usage

### 1. Create a Python virtual environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the trained model and preprocessing files in the `model/` directory.

### 4. Run the server
```bash
python app.py
```

### 5. Access the API
Open your browser or API client at: [http://localhost:8080/](http://localhost:8080/)

## Requirements
- Python 3.8+
- Flask
- TensorFlow/Keras
- pandas, numpy, joblib, holidays, requests, openmeteo_requests, requests_cache, retry_requests