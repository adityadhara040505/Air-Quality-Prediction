import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprcessing import MinMaxScaler

def get_project_root():
    """Get the root directory of the project"""
    current_dir = os.path.dirname(os.path.abspath(_file_))
    return os.path.dirname(current_dir)

def load_artifacts():
    """Load the trained model and scaler"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    model = load_model(os.path.join(models_dir, 'aqi_predictor.h5'))
    scaler = joblib.load(os.path.join(models_dir, 'aqi_scaler.pkl'))
    return model, scaler

def get_user_input():
    """Collect input features from the user with date input"""
    print("\n" + "="*50)
    print("Please enter the following details:")
    print("="*50)
    
    # Get date input
    date_str = input("Date (YYYY-MM-DD): ")
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    features = {
        'date': input_date,
        'CO': float(input("CO level (ppm): ")),
        'Ozone': float(input("Ozone level (ppb): ")),
        'PM10': float(input("PM10 level (μg/m³): ")),
        'PM25': float(input("PM2.5 level (μg/m³): ")),
        'NO2': float(input("NO2 level (ppb): ")),
        'AQI_lag1': float(input("Previous day's AQI: ")),
        'AQI_lag7': float(input("AQI from 7 days ago: ")),
        'AQI_rolling7_mean': float(input("7-day average AQI: ")),
        'AQI_rolling30_mean': float(input("30-day average AQI: "))
    }
    
    # Calculate derived temporal features
    features['DayOfYear'] = input_date.timetuple().tm_yday
    features['Month'] = input_date.month
    features['DayOfWeek'] = input_date.weekday()  # Monday=0
    
    return features

def prepare_input(features, scaler):
    """Prepare input data for prediction"""
    input_df = pd.DataFrame([features])
    
    feature_cols = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2',
                   'DayOfYear', 'Month', 'DayOfWeek',
                   'AQI_lag1', 'AQI_lag7',
                   'AQI_rolling7_mean', 'AQI_rolling30_mean']
    
    cols_to_normalize = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2',
                        'AQI_lag1', 'AQI_lag7',
                        'AQI_rolling7_mean', 'AQI_rolling30_mean']
    
    input_df[cols_to_normalize] = scaler.transform(input_df[cols_to_normalize])
    input_array = input_df[feature_cols].values.reshape(1, 1, -1)
    
    return input_array, input_df

def save_prediction(prediction_data, filename="aqi_predictions.csv"):
    """Save prediction results to CSV file"""
    project_root = get_project_root()
    output_dir = os.path.join(project_root, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to DataFrame if not already
    if not isinstance(prediction_data, pd.DataFrame):
        prediction_data = pd.DataFrame([prediction_data])
    
    # Write to CSV (append if file exists)
    if os.path.exists(filepath):
        prediction_data.to_csv(filepath, mode='a', header=False, index=False)
    else:
        prediction_data.to_csv(filepath, index=False)
    
    print(f"\nPrediction saved to: {filepath}")

def plot_aqi_history(predictions_file="aqi_predictions.csv"):
    """Plot historical predictions"""
    project_root = get_project_root()
    filepath = os.path.join(project_root, 'predictions', predictions_file)
    
    if not os.path.exists(filepath):
        print("No prediction history found")
        return
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['predicted_aqi'], 'b-', label='Predicted AQI')
    
    # Add AQI category reference lines
    plt.axhline(y=50, color='g', linestyle='--', alpha=0.3, label='Good')
    plt.axhline(y=100, color='y', linestyle='--', alpha=0.3, label='Moderate')
    plt.axhline(y=150, color='orange', linestyle='--', alpha=0.3, label='Unhealthy for Sensitive Groups')
    plt.axhline(y=200, color='r', linestyle='--', alpha=0.3, label='Unhealthy')
    plt.axhline(y=300, color='purple', linestyle='--', alpha=0.3, label='Very Unhealthy')
    
    plt.title('AQI Prediction History')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(project_root, 'predictions', 'aqi_history.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

def predict_aqi(model, scaler, input_features):
    """Make a prediction using the trained model"""
    prepared_input, input_df = prepare_input(input_features, scaler)
    prediction = model.predict(prepared_input)
    return prediction[0][0], input_df

def get_aqi_category(aqi):
    """Get AQI category description"""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def main():
    print("Air Quality Index (AQI) Prediction System")
    print("="*50)
    
    # Load model
    print("\nLoading prediction model...")
    model, scaler = load_artifacts()
    
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            # Make prediction
            predicted_aqi, input_df = predict_aqi(model, scaler, user_input)
            category = get_aqi_category(predicted_aqi)
            
            # Prepare results dictionary
            result = {
                'date': user_input['date'].strftime('%Y-%m-%d'),
                'predicted_aqi': predicted_aqi,
                'aqi_category': category,
                **{k: v for k, v in user_input.items() if k != 'date'}
            }
            
            # Display result
            print("\n" + "="*50)
            print(f"Predicted AQI: {predicted_aqi:.2f} ({category})")
            print("="*50)
            # Save prediction
            save_prediction(result)
            
            # Show history plot
            plot_aqi_history()
            
            # Continue?
            cont = input("\nMake another prediction? (y/n): ").lower()
            if cont != 'y':
                print("\nThank you for using the AQI Prediction System!")
                break
                
        except ValueError as ve:
            print(f"Invalid input! {str(ve)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

if _name_ == "_main_":
    main()