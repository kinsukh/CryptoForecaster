import streamlit as st
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

def lstm_prediction(ticker, start_date, end_date, load_var, save_var):
    from Lstm_model import LSTMPredictor
    from configs.LstmConfig import Config

    if load_var:
        try:
            predictor = LSTMPredictor.load_model(Config.MODEL_SAVE_PATH)
            st.success("Model loaded successfully.")
        except FileNotFoundError:
            st.error("Model file not found. Please train a new model.")
            return

        df = predictor.yf_Down(ticker, start_date, end_date)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_pred = predictor.predict(X_test)
        y_true = predictor.scaler_y.inverse_transform(y_test)
        rmse = predictor.evaluate_model(y_true, y_pred)
        st.info(f'RMSE: {rmse}')

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - LSTM Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

    else:
        predictor = LSTMPredictor()
        Config.TICKER = ticker
        df = predictor.yf_Down(Config.TICKER, Config.START_DATE, Config.END_DATE)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Training model with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def train_with_progress(X_train, y_train, X_test, y_test):
            history = predictor.train_model(X_train, y_train, X_test, y_test)
            for i in range(len(history.history['loss'])):
                progress = (i + 1) / len(history.history['loss'])
                progress_bar.progress(progress)
                status_text.text(f"Training progress: {progress*100:.2f}%")
            return history

        history = train_with_progress(X_train, y_train, X_test, y_test)

        y_pred = predictor.predict(X_test)
        y_true = predictor.scaler_y.inverse_transform(y_test)
        rmse = predictor.evaluate_model(y_true, y_pred)
        st.info(f'RMSE: {rmse}')

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - LSTM Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        if save_var:
            predictor.save_model(Config.MODEL_SAVE_PATH)
            st.success("Model saved successfully.")

def xgboost_prediction(ticker, start_date, end_date, load_var, save_var):
    from Xgboost_model import XGBoost_Predictor

    config_path = "configs/Xgboost_config.yaml"

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        config = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "test_size": 0.2,
            "plot_dir": "plots",
            "hyperparameter_tuning": {
                "max_depth": {"min": 3, "max": 10},
                "n_estimators": {"min": 100, "max": 500},
                "learning_rate": {"min": 0.01, "max": 0.3},
                "subsample": {"min": 0.8, "max": 1.0},
                "colsample_bytree": {"min": 0.8, "max": 1.0},
                "n_iter": 20
            }
        }

    config['start_date'] = start_date
    config['end_date'] = end_date
    config['ticker'] = ticker

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    predictor = XGBoost_Predictor(config_path)

    if load_var:
        model_path = "models/xgboost_model.json"
        try:
            predictor.load_model(model_path)
            st.success("Model loaded successfully.")
        except FileNotFoundError:
            st.error("Model file not found. Please train a new model.")
            return

        df = predictor.download_data()
        X_test, _, _, _ = predictor.prepare_data(df)

        y_pred = predictor.predict(X_test)
        y_pred_real = predictor.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(df['Close'].values[-len(y_pred_real):], label='Actual')
        ax.plot(y_pred_real, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - XGBoost Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
        st.success("Prediction completed.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def run_with_progress():
            df = predictor.download_data()
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            
            progress_bar.progress(0.2)
            status_text.text("Data prepared. Optimizing hyperparameters...")
            
            best_params = predictor.optimize_xgb(X_train, y_train)
            
            progress_bar.progress(0.6)
            status_text.text("Hyperparameters optimized. Training model...")
            
            predictor.train_model(X_train, y_train, best_params)
            
            progress_bar.progress(0.8)
            status_text.text("Model trained. Evaluating...")
            
            y_pred = predictor.predict(X_test)
            rmse, y_test_real, y_pred_real = predictor.evaluate(y_test, y_pred)
            
            progress_bar.progress(1.0)
            status_text.text("Evaluation complete!")
            
            return y_test_real, y_pred_real, rmse

        y_test_real, y_pred_real, rmse = run_with_progress()

        st.info(f'RMSE: {rmse}')

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(y_test_real, label='Actual')
        ax.plot(y_pred_real, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - XGBoost Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        if save_var:
            model_filename = "models/xgboost_model.json"
            predictor.save_model(model_filename)
            st.success(f"Model saved to {model_filename}.")

def random_forest_prediction(ticker, start_date, end_date, load_var, save_var):
    from Random_Forest_Regressor import RandomForestPredictor
    rf_predictor = RandomForestPredictor()

    if load_var:
        try:
            rf_predictor.load_model()
            st.success("Model loaded successfully.")
        except FileNotFoundError:
            st.error("Model file not found. Please train a new model.")
            return

        mse, mae, r2, dates, y_true, y_pred = rf_predictor.predict_new_data(ticker, start_date, end_date)
        st.success(f'MSE: {mse}\nMAE: {mae}\nR-squared: {r2}')
        
        # Plot results
        fig, ax = plt.subplots()
        ax.plot(dates, y_true, label='Actual')
        ax.plot(dates, y_pred, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - Random Forest Model')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def run_with_progress():
            status_text.text("Downloading and preparing data...")
            progress_bar.progress(0.1)
            
            df = rf_predictor.download_and_prepare_data(ticker, start_date, end_date)
            X, y = rf_predictor.prepare_features_and_target(df)
            
            status_text.text("Training and evaluating model...")
            progress_bar.progress(0.3)
            
            X_test, y_test, predictions, mse, mae, r2 = rf_predictor.train_and_evaluate_model(X, y)
            
            status_text.text("Model training complete!")
            progress_bar.progress(1.0)
            
            return X_test, y_test, predictions, mse, mae, r2

        X_test, y_test, predictions, mse, mae, r2 = run_with_progress()

        st.success(f'MSE: {mse}\nMAE: {mae}\nR-squared: {r2}')
        
        # Plot results
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test, label='Actual')
        ax.plot(y_test.index, predictions, label='Predicted')
        ax.set_title(f'{ticker} Price Prediction - Random Forest Model')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
        
        if save_var:
            rf_predictor.save_model()
            st.success("Model saved successfully.")

# Main Application Logic

st.title("CryptoForcaster")

# User Inputs
ticker = st.text_input("Ticker:", value="BTC-USD")
start_date = st.date_input("Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime("2024-01-01"))

# Load/Save Options
load_var = st.checkbox("Load Model")
save_var = st.checkbox("Save Model")

# Model Selection
model_selection = st.radio("Select Model:", options=["LSTM", "XGBoost", "Random Forest"])

# Run Button
if st.button("Run"):
    if model_selection == "LSTM":
        lstm_prediction(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), load_var, save_var)
    elif model_selection == "XGBoost":
        xgboost_prediction(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), load_var, save_var)
    elif model_selection == "Random Forest":
        random_forest_prediction(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), load_var, save_var)

st.sidebar.title("About")
st.sidebar.info("This application predicts cryptocurrency and stock prices using various machine learning models.")
