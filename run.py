#! /usr/bin/python

from data_cleaning import data_cleaner
from feature_engineering import feature_engineer
from forecasting import arima_forecaster, xgbregressor_forecaster

def run_pipeline():
    cleaner = data_cleaner.DataCleaner()
    cleaner.run()

    engineer = feature_engineer.FeatureEngineer(username=cleaner.username, crop_name='UPCN')
    engineer.run()
    processed_data = engineer.binned_subsidies_df

    arima = arima_forecaster.ARIMAForecaster(processed_data)
    arima.run()

    xgb = xgbregressor_forecaster.XGBRegressorForecaster(processed_data)
    xgb.run()

if __name__ == '__main__':
    run_pipeline()
