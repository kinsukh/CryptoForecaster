# CryptoForcaster
Crypto & Stock Price Prediction and Forecasting.

The project let's you select model, start & end date, available models are: Lstm, XGBoost and Random Forest.

### Dependencies:

```
pip install tensorflow yfinance ta scikit-learn pandas numpy matplotlib tkinter pickle PyYAML bayesian-optimization
```
In case of an update breaks something:
- tensorflow = 2.16.1
- keras = 3.3.3
- yfinance = 0.2.40
- ta = 0.11.0
- scikit-learn = 1.5.0

## LSTM
Long Short-Term Memory (LSTM). It leverages historical price data from Yahoo Finance and incorporates technical indicators (SMA, EMA, RSI) as well as time-based features to improve the prediction accuracy.

## XGBoost
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. 
