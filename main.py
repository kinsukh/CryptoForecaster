import numpy as np
import logging

def main():
    ticker = 'BTC-USD'

    print("1.LSTM")
    print("2.Xgboost")
    print("3.Random Forest")
    selection = input("Select Model:    ")

    if selection == "1":
        # LSTM
        from Lstm_model import LSTMPredictor
        LSTMPredictor.run(ticker)

    elif selection == "2":
        #XGBoost
        from Xgboost_model import XGBoost_Predictor
        import yaml

        config_path = "configs/Xgboost_config.yaml"

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            config = {
                "ticker": ticker,
                "start_date": "",
                "end_date": "",
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
        print("XgboostRegressor selected.")
        print("Load saved model? (Must be in models folder.)")
        selection_c = input("Y/N: ").upper()

        if selection_c == "Y":
            model_path = "models/xgboost_model.json"
            predictor = XGBoost_Predictor(config_path)
            predictor.load_model(model_path)
            print("Model loaded successfully.")

            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
            config['ticker'] = ticker

            # Use loaded model for prediction
            df = predictor.download_data()
            X_test, _, _, _ = predictor.prepare_data(df)

            y_pred = predictor.predict(X_test)
            y_pred_real = predictor.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

            # Plot results
            predictor.plot_results(df['Close'].values[-len(y_pred_real):], y_pred_real)

            print("Prediction completed. Check the plots directory for visualization.")

        elif selection_c == "N":
            config['start_date'] = input("Start Date (YYYY-MM-DD): ")
            config['end_date'] = input("End Date (YYYY-MM-DD): ")
            config['ticker'] = ticker

            # Save updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

            # Create and run the predictor
            predictor = XGBoost_Predictor(config_path)
            predictor.run()

            save_model = input("Save model? Y/N: ").upper()
            if save_model == "Y":
                model_filename = "models/xgboost_model.json"
                predictor.save_model(model_filename)
                print(f"Model saved to {model_filename}.")
        else:
            print("Invalid selection.")

    elif selection == "3":
        # Random Forest
        from Random_Forest_Regressor import RandomForestPredictor
        print("Random Forest selected.")
        rf_predictor = RandomForestPredictor()

        print("Load saved model? (Must be in the 'models' directory.)")
        selection_c = input("Y/N: ")
        if selection_c.upper() == "Y":
            rf_predictor.load_model()
            print(f"Model for {ticker} loaded successfully.")

            start_date = input("Start Date for new data (YYYY-MM-DD): ")
            end_date = input("End Date for new data (YYYY-MM-DD): ")
            mse, mae, r2, dates, y_true, y_pred = rf_predictor.predict_new_data(ticker, start_date, end_date)
            print(f'MSE on new data: {mse}')
            print(f'MAE on new data: {mae}')
            print(f'R-squared on new data: {r2}')
            print(f'Predictions saved in plots/{ticker}_prediction_new_data.png')
            # Option to display some of the predictions
            show_predictions = input("Do you want to see some of the predictions? Y/N: ")
            if show_predictions.upper() == "Y":
                num_predictions = min(30, len(dates))  # Show up to 30 predictions
                print("\nSample predictions:")
                print("Date\t\tActual Price\tPredicted Price")
                for i in range(num_predictions):
                    print(f"{dates[i]:%Y-%m-%d}\t{y_true[i][0]:.2f}\t\t{y_pred[i][0]:.2f}")

        elif selection_c.upper() == "N":
            start_date = input("Start Date (YYYY-MM-DD): ")
            end_date = input("End Date (YYYY-MM-DD): ")
            mse, mae, r2 = rf_predictor.run(ticker, start_date, end_date)
            print(f'MSE: {mse}')
            print(f'MAE: {mae}')
            print(f'R-squared: {r2}')
            savemodel = input("Save model? Y/N: ")
            if savemodel.upper() == "Y":
                rf_predictor.save_model()
                print(f"Model for {ticker} saved successfully.")
            else:
                print("Model not saved.")
        else:
            print("Invalid input.")
    
    else:
        print('Incorrect Input')

if __name__ == "__main__":
    main()
