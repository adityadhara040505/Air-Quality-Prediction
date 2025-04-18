import os
import warnings
warnings.filterwarnings('ignore')

# Data Science Libraries
import pandas as pd
import numpy as np

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import joblib

def get_project_root():
    """Get the root directory of the project (Air-Quality-Prediction folder)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one level from notebooks folder

def clean_data(df):
    """Clean and preprocess the data"""
    # Drop unnecessary columns
    df.drop(columns=['Site Name (of Overall AQI)', 'Site ID (of Overall AQI)', 
                    'Source (of Overall AQI)', 'Main Pollutant'], inplace=True, errors='ignore')
    
    # Convert pollutant columns to numeric
    pollutant_cols = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2']
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace '-' with NaN
    df.replace('-', np.nan, inplace=True)
    
    # Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df
def add_features(df):
    """Add temporal and statistical features"""
    # Temporal features
    df['DayOfYear'] = df.index.dayofyear
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    
    # Lag features
    df['AQI_lag1'] = df['Overall AQI Value'].shift(1)
    df['AQI_lag7'] = df['Overall AQI Value'].shift(7)
    
    # Rolling statistics
    df['AQI_rolling7_mean'] = df['Overall AQI Value'].rolling(7).mean()
    df['AQI_rolling30_mean'] = df['Overall AQI Value'].rolling(30).mean()
    
    # Drop rows with NaN values created by lag/rolling features
    df.dropna(inplace=True)
    
    return df

def normalize_data(df):
    """Normalize the data using MinMaxScaler"""
    scaler = MinMaxScaler()
    
    # Select columns to normalize (excluding temporal features)
    cols_to_normalize = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2', 
                        'AQI_lag1', 'AQI_lag7', 
                        'AQI_rolling7_mean', 'AQI_rolling30_mean']
    
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    return df, scaler

def create_sequences(data, target_col, window_size=30):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - window_size):
        # Get window_size days of features
        window = data.iloc[i:(i + window_size)].drop(columns=[target_col])
        # Get the next day's target value
        target = data.iloc[i + window_size][target_col]
        X.append(window.values)
        y.append(target)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build the LSTM model architecture"""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.001),
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual AQI', alpha=0.7)
    plt.plot(y_pred, label='Predicted AQI', alpha=0.7)
    plt.title('Actual vs Predicted AQI Values')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI Values')
    plt.show()

def predict_future(model, last_window, future_steps=30):
    """
    Predict future AQI values using the trained model
    
    Args:
        model: Trained LSTM model
        last_window: Last window of observed data (shape: [1, window_size, n_features])
        future_steps: Number of future steps to predict
        
    Returns:
        Array of predicted values
    """
    predictions = []
    current_window = last_window.copy()
    
    for _ in range(future_steps):
        # Predict next step
        next_pred = model.predict(current_window, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update window with prediction
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1, 0] = next_pred  # Update AQI value
        
    return np.array(predictions)

def main():
    # Set up paths
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'notebooks', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        # Load individual year files
        df_2023 = pd.read_csv(os.path.join(data_dir, 'aqidaily2023.csv'))
        df_2024 = pd.read_csv(os.path.join(data_dir, 'aqidaily2024.csv'))
        df_2025 = pd.read_csv(os.path.join(data_dir, 'aqidaily2025.csv'))
        
        # Combine into one DataFrame
        df = pd.concat([df_2023, df_2024, df_2025])
        
        # Convert date column and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Data preprocessing
    print("Preprocessing data...")
    df = clean_data(df)
    df = add_features(df)
    
    # Normalize data
    df, scaler = normalize_data(df)
    
    # Prepare sequences
    print("Creating sequences...")
    X, y = create_sequences(df, 'Overall AQI Value', 30)
    
    # Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, shuffle=False)
    
    # Build and train model
    print("Building and training model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss'
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    model = load_model(os.path.join(models_dir, 'best_model.h5'))
    evaluate_model(model, X_test, y_test)
    
    # Make future predictions
    print("\nMaking future predictions...")
    last_window = X_test[-1:]
    future_predictions = predict_future(model, last_window, 30)
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_test[-30:])), y_test[-30:], label='Last 30 Actual')
    plt.plot(np.arange(len(y_test[-30:]), len(y_test[-30:]) + 30), future_predictions, 
             label='Next 30 Predicted', color='orange')
    plt.title('Future AQI Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.show()
    
    # Save final artifacts
    print("\nSaving model artifacts...")
    model.save(os.path.join(models_dir, 'aqi_predictor.h5'))
    joblib.dump(scaler, os.path.join(models_dir, 'aqi_scaler.pkl'))
    print("Model and scaler saved successfully at:", models_dir)

if __name__ == "__main__":
    main()