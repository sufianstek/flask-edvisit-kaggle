from flask import Flask, request, jsonify, url_for
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import timedelta
import requests
import openmeteo_requests
import json
import holidays
import requests_cache
from retry_requests import retry

app = Flask(__name__)

model_path = 'model/'

# --- Load the Saved Model, Preprocessor, and Config Info ---
model = load_model(f'{model_path}forecast_model.keras', compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
preprocessor = joblib.load(f'{model_path}forecast_model_preprocessor.pkl')
model_info = joblib.load(f'{model_path}forecast_model_model_info.pkl')

print(model_info)

N_STEPS = model_info['n_steps']
numerical_features = model_info['numerical_features']
categorical_features = model_info['categorical_features']
feature_names = model_info['feature_names']

# --- Forecasting Logic ---

def get_meteo():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Open-Meteo forecast request for Kuantan, Pahang
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 3.8077,
        "longitude": 103.326,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
        "timezone": "Asia/Singapore",
	    "past_days": 14,
        "forecast_days": 14
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process response
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    return response




@app.route('/', methods=['GET'])
def index():

    response =get_meteo()

    # Extract daily weather data
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_relative_humidity_2m_mean = daily.Variables(1).ValuesAsNumpy()

    # Create date range
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )

    # Create DataFrame
    daily_dataframe = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": daily_temperature_2m_mean,
        "relative_humidity_2m_mean": daily_relative_humidity_2m_mean
    })

    # Define Malaysian national holidays for relevant years
    years = daily_dataframe["date"].dt.year.unique()
    malaysia_holidays = holidays.MY(years=years)

    # Add known Pahang state holidays manually
    pahang_state_holidays = {
        "2025-05-22": "Hari Hol Pahang",
        "2025-07-30": "Hari Keputeraan Sultan Pahang"   
    }
    for date_str, name in pahang_state_holidays.items():
        malaysia_holidays[date_str] = name

    # Annotate holidays in the DataFrame
    daily_dataframe["is_holiday"] = daily_dataframe["date"].dt.date.apply(lambda x: x in malaysia_holidays)

    # Output
    print(daily_dataframe)

    # Generate input data sequence with correct column names
    sequence = []
    for _, row in daily_dataframe.iterrows():
        date = row["date"]
        sequence.append({
            'month': date.month,
            'day_of_week': date.weekday(),  # Monday=0
            'mean_temperature': round(row["temperature_2m_mean"], 2),
            'mean_relative_humidity': round(row["relative_humidity_2m_mean"], 2),
            'holiday': int(row["is_holiday"]),  # Ensure it's int (0 or 1)
        })

    # Use the first date in the forecast as start_date
    start_date = daily_dataframe["date"].iloc[0]

    # Send POST request
    url = request.base_url+'forecast'

    payload = {
        'sequence': sequence,
        'start_date': start_date.strftime('%Y-%m-%d')
    }
    response = requests.post(url, json=payload)

    # Output the results
    if response.status_code == 200:
        results = response.json()
        
        # Prepare the formatted list
        output = []
        for date_str, pred, temp, humidity in zip(results['forecast_dates'], results['predicted_ed_visits'], sequence, sequence ):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_name = date_obj.strftime('%A')
            is_weekend = date_obj.weekday() in [5, 6]  # Saturday=5, Sunday=6
            is_holiday = date_obj in malaysia_holidays
            output.append({
                "date": date_str,
                "day": day_name,
                "non_working_day": is_weekend or is_holiday,
                "predicted_ed_visits": round(pred),  # rounding for neatness
                "avg_temp": round(temp['mean_temperature'],1),
                "avg_humidity": round(humidity['mean_relative_humidity'],2)
            })

        # Print JSON output
        return jsonify(output)
    else:
        return jsonify("Error:", response.status_code, response.text)


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        raw_sequence = data['sequence']

        if not isinstance(raw_sequence, list) or len(raw_sequence) < 14:
            return jsonify({'error': "At least 14 days of input required"}), 400

        # Use only the last 14 days as initial input
        sequence_df = pd.DataFrame(raw_sequence[-14:])
        last_date = pd.Timestamp(data.get('start_date', pd.Timestamp.today().strftime('%Y-%m-%d')))

        X_processed = preprocessor.transform(sequence_df).toarray()
        current_sequence = X_processed.reshape(1, 14, -1)

        forecasts = []
        forecast_dates = []

        for i in range(28):  # 28-day forecast
            # Predict
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            forecasts.append(round(float(next_pred), 2))

            # Advance date
            next_date = last_date + timedelta(days=1)
            forecast_dates.append(str(next_date.date()))
            last_date = next_date

            # Prepare new input row
            last_input = sequence_df.iloc[-1].copy()
            last_input['month'] = next_date.month
            last_input['day_of_week'] = next_date.dayofweek
            if 'holiday' in last_input:
                last_input['holiday'] = 0  # Or check against real calendar

            # Create DataFrame and transform
            next_df = pd.DataFrame([last_input], columns=sequence_df.columns)
            next_processed = preprocessor.transform(next_df).toarray()

            # Slide: remove oldest, add newest
            current_sequence = np.concatenate([
                current_sequence[0, 1:, :],
                next_processed.reshape(1, -1)
            ]).reshape(1, 14, -1)

            # Update raw input DataFrame
            sequence_df = pd.concat([sequence_df.iloc[1:], next_df], ignore_index=True)

        return jsonify({
            'forecast_dates': forecast_dates,
            'predicted_ed_visits': forecasts
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
    #app.run(debug=True, host="127.0.0.1", port=5000)
