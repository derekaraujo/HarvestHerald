import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import copy
import os
import warnings
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6

class XGBRegressorForecaster:
    '''
    A class to run an XGBRegressor boosted regression tree on the pre-processed data frame
    generated by feature_engineering.py.  A dynamic walk-forward forecast is generated by
    calling the method XGBRegressorForecaster.walk_forward_forecast(): at each step the model
    forecasts the target variable one time sample into the future and adds the true, observed 
    target variable to the training set.
        Inputs:
            - dataframe: the Pandas data frame of engineered features generated by 
              feature_engineering.py, stored as the binned_subsidies_df attribute of the
              FeatureEngineer class.
            - target_column (string): the column name of the target variable in the dataframe 
              to be forecast.
            - train_columns (list): a list of column names in the dataframe to be used as traning 
              features.  If unspecified, the default value is None and a list of default training 
              features is generated during initialization.
            - shift_target (bool): toggles the option to use the feature variables during each 
              month to train on/predict the target variable in the same month, versus one
              month in the future.  If set to False the features in each monthly bin are trained
              on the target variable in the same bin; a forecast of the target variable one month
              in the future is generated using the feature variables in the future month's bin.
              Setting shift_target to True operates the model on the more realistic assumption 
              that the future month's feature variables are unnkown at the time of forecasting: the 
              features in each monthly bin are trained on the target variable one month in the 
              future, and a forecast of the future month's target variable is generated using the 
              feature variables as measured in the present.
    '''
    
    def __init__(self, dataframe, target_column='mean_transaction_amount', train_columns=None,
                 shift_target=True):
        self.n_estimators = 100
        self.max_depth = 20
        self.dataframe = dataframe
        self.shift_target = shift_target
        self.target_column = target_column
        if train_columns != None:
            self.train_columns = train_columns
        else:
            self.train_columns = [col for col in self.dataframe.columns if 
                                  '_customer_number_startswith_A' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_sin_day_of_year' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_cos_day_of_year' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_sin_day_of_month' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_cos_day_of_month' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_sin_season' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_cos_season' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_program_code' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_program_year' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_state_code' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_county_code' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_days_of_drought' in col]
            self.train_columns += [col for col in self.dataframe.columns if '_drought_severity' in col]
            if 'mean_futures_30day_std' in self.dataframe.columns:
                self.train_columns += [col for col in self.dataframe.columns if '_futures' in col]
    
    def form_training_and_test_sets(self, training_fraction=0.66):
        '''
        A method to form training and test sets from the input dataframe.
        By default the fraction of data in the training set (training_fraction)
        is 0.66, with the final 0.34 of the time-ordered data resderved for 
        testing via the walk_forward_forecast() function defined below.
        '''
        self.X = self.dataframe[self.train_columns].values
        self.y = self.dataframe[self.target_column].values
        # reserve final third of the time-ordered data for testing
        self.split_idx = int(round(len(self.y) * training_fraction))
        self.X_train = self.X[:self.split_idx, :]
        self.X_test = self.X[self.split_idx:, :]
        self.y_train = self.y[:self.split_idx]
        self.y_test = self.y[self.split_idx:]
    
    # Hyperparameter grid search functions:
    
    def evaluate_xgb_model(self, max_depth, n_estimators):
        '''
        A helper function for the do_hyperparameter_grid_search() function. 
        Given specified n_estimator and max_depth hyperparameters, it tests 
        an XGBRegressor model tuned to the speficied hyperparameters.
        Output: the root mean squared error for the model.
        '''
        predictions = []
        X, y = self.X_train, self.y_train
        split_idx = int(round(0.66 * X.shape[0]))
        for i in range(len(y[split_idx:])):
            model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators)
            if self.shift_target:
                model.fit(X[:split_idx + i - 1, :], y[1:split_idx + i])
                output = model.predict(X[split_idx + i - 1, :].reshape((1,-1)))
            else:
                model.fit(X[:split_idx + i, :], y[:split_idx + i])
                output = model.predict(X[split_idx + i, :].reshape((1,-1)))
            prediction = output[0]
            predictions.append(prediction)
        mse = mean_squared_error(y[split_idx:], predictions) # DEBUG
        return np.sqrt(mse)
    
    def do_hyperparameter_grid_search(self, verbose=True):
        '''
        A method to perform a hyperparameter grid search to identify the
        optimal n_estimators and max_depth hyperparameters for the XGBRegressor 
        model.  It calls the helper function evaluate_xgb_model() to test each model,
        and assigns the optimal hyperparameters to self.n_estimators and self.max_depth.
        '''
        n_estimators_list = [5,10,20,40,60,100,200,300,500]
        sqrt_n_features = int(round(np.sqrt(self.X_train.shape[1])))
        max_depths_list = range(sqrt_n_features - 5, sqrt_n_features + 5)
        
        warnings.filterwarnings("ignore")
        best_rmse, best_params = np.inf, None
        
        for max_depth in max_depths_list:
            for n_estimators in n_estimators_list:
                if verbose:
                    print 'Evaluating max_depth=%s, n_estimators=%s' % (max_depth, n_estimators)
                rmse = self.evaluate_xgb_model(max_depth, n_estimators)
                if rmse < best_rmse:
                    best_rmse, best_params = rmse, [max_depth, n_estimators]
                if verbose:
                    print('XGB%s RMSE=%s' % (best_params, rmse))
        if verbose:
            print 'Best XGB: [max_depth, n_estimators]=%s, RMSE=%s' % (best_params, best_rmse)
            
        self.max_depth = best_params[0]
        self.n_estimators = best_params[1]
    
    # Forecasting functions:

    def walk_forward_forecast(self, n_estimators=None, max_depth=None): 
        '''
        A method to perform a walk-forward forecast of the test data set.
        At each step the regressor produces a forecast of the target column
        one time sample in the future; records the error = (predicted value 
        minus observed value); and adds the observed value to the training set
        for use in the next forecasting step.
        '''
        if n_estimators == None:
            n_estimators = self.n_estimators
        if max_depth == None:
            max_depth = self.max_depth
        predictions = []
        print 'Shape:', self.X.shape
        for i in range(len(self.y_test)):
            model = XGBRegressor(n_estimators=self.n_estimators, max_depth=max_depth)
            if self.shift_target:
                model.fit(self.X[:self.split_idx + i - 1, :], self.y[1:self.split_idx + i])
                output = model.predict(self.X[self.split_idx + i - 1, :].reshape((1,-1)))
            else:
                model.fit(self.X[:self.split_idx + i, :], self.y[:self.split_idx + i])
                output = model.predict(self.X[self.split_idx + i, :].reshape((1,-1)))
            prediction = output[0]
            predictions.append(prediction)
        error = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(error)
        print 'Test RMSE: %.3f' % rmse
        self.rmse = rmse
        self.model = model
        self.predictions = predictions

    def get_feature_importances(self, model=None):
        '''
        A method to obtain the feature importances identified by the regression model.
        Output:
            - importances = the fractional importance of each predictor
            - names = the names of the columns corresponding to each predictor in the
              self.dataframe data set
            - indices = the indices of the importances, sorted from largest to smallest 
              fractional importance.
        '''
        if model == None:
            model = self.model
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = np.array(self.train_columns)[indices]
        return names, importances, indices
    
    def generate_bootstrap_confidence_intervals(self, alpha=0.95, n_iterations=10): 
        '''
        A method to generate bootstrap confidence intervals for the regressor predictions
        by iteratively excluding data points, calculating a range of predictions, and 
        reporting the cutoff values for the 5th and 95th percentile predictions. 
          - The size of the confidence interval can be adjusted by changing alpha (e.g.,  
            set alpha=0.9 to obtain the bounds for a 90% confidence interval).
        Output: 
          - lowers = array of N lower bound values for the N model predictions
          - uppers = array of N upper bound values for the N model predictions
        '''
        uppers, lowers = [], []
        for i in range(len(self.y_test)):
            bootstrap_predictions = []
            for iteration in range(n_iterations):
                if self.shift_target:
                    X_train = self.X[:self.split_idx + i - 1, :]
                    y_train = self.y[1:self.split_idx + i]
                    X_test = self.X[self.split_idx + i - 1, :]
                else:
                    X_train = self.X[:self.split_idx + i, :]
                    y_train = self.y[:self.split_idx + i]
                    X_test = self.X[self.split_idx + i, :]
                indices = range(len(y_train))
                n_size = int(round(0.3*len(indices)))
                indices_to_exclude = np.unique(np.random.choice(indices, size=n_size))
                train_idxs = np.array([idx for idx in indices if idx not in indices_to_exclude])
                model = XGBRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators)
                model.fit(X_train[train_idxs,:], y_train[train_idxs])
                output = model.predict(X_test.reshape((1,-1)))
                prediction = output[0]
                bootstrap_predictions.append(prediction)

            p = ((1. - alpha)/2.) * 100.
            lower = np.percentile(bootstrap_predictions, p)
            lowers.append(lower)

            p = (alpha + ((1. - alpha)/2.)) * 100.
            upper = np.percentile(bootstrap_predictions, p)
            uppers.append(upper)
        return np.array(lowers), np.array(uppers)
    
    # Plotting functions:

    def plot_forecast(self, lower_conf_intervals=[], upper_conf_intervals=[],
                      show=True, save=False):
        '''
        A method to plot the forecast generated by the walk_forward_forecast() function.
        To plot a shaded confidence interval, pass the output of the 
        generate_bootstrap_confidence_intervals() function as lower_conf_intervals 
        and upper_conf_intervals.
        '''
        xs = self.dataframe.mean_transaction_yearmonth.values
        xs = pd.to_datetime(xs)
        predictions = self.predictions
        ylabel = 'Monthly Avg Subsidy (USD)'
        if self.target_column == 'mean_transaction_amount':
            ys = self.dataframe.mean_transaction_amount
            rmse = self.rmse
        elif self.target_column == 'log_mean_transaction_amount':
            ys = df.log_mean_transaction_amount
            ylabel = 'log Monthly Avg Subsidy (log USD)'
            mse = mean_squared_error(y_test, np.exp(predictions))
            rmse = np.sqrt(mse)
        pl.plot(xs[:self.split_idx], ys[:self.split_idx], 'bo-', alpha=0.5, label='train')
        pl.plot(xs[self.split_idx:], ys[self.split_idx:], 'co-', alpha=1.0, label='test')
        pl.plot(xs[self.split_idx:], predictions, 'ro-', alpha=0.5, label='prediction')
        if len(lower_conf_intervals) > 0 and len(upper_conf_intervals) > 0:
            pl.fill_between(xs[self.split_idx:], lower_conf_intervals, upper_conf_intervals, 
                            color='r', alpha=0.2, label='95% prediction interval')
        pl.title('XGB Regressor Forecast: RMSE = %.2f' % rmse)
        pl.ylabel(ylabel)    
        pl.legend()
        pl.xticks(rotation=30, ha='right')
        if save:
            pl.savefig('XGB_regressor_forecast.png')
        if show:
            pl.show()
        
    def plot_feature_importances(self, names, importances, indices, show=True, save=False):
        '''
        A method to plot the feature importances output by the get_feature_importances() 
        method.
        '''
        pl.figure(figsize=(14,5))
        pl.title("Feature Importances (Top 10)")
        sns.set_context("poster", font_scale=1.5)
        sns.set_color_codes("muted")
        sns.barplot(importances[indices][:10], names[:10], palette='Blues_d')
        if save:
            pl.savefig('feature_importances.png')
        if show:
            pl.show()
    
    # Wrapper function:
    
    def run(self):
        '''
        A wrapper function to run the forecasting pipeline.
        '''
        self.form_training_and_test_sets()
        self.do_hyperparameter_grid_search()
        self.walk_forward_forecast()
        names, importances, indices = self.get_feature_importances()
        self.plot_feature_importances(names, importances, indices)
        lower_conf_intervals, upper_conf_intervals = self.generate_bootstrap_confidence_intervals()
        self.plot_forecast(lower_conf_intervals=lower_conf_intervals, 
                           upper_conf_intervals=upper_conf_intervals)