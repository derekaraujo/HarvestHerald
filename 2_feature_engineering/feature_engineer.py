import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
import seaborn as sns
import sklearn
import psycopg2
from datetime import timedelta
from sklearn.cluster import KMeans
import nltk
import re


class FeatureEngineer:

    '''
    A class to engineer features from the subsidy payment, weather, and commodities futures data
    sets, as follows:
        - Load the subsidy payment, weather, and commodity futures data sets
        - For each subsidy payment record: engineer new features using the three
          data sets and the 'add_[...]' functions defined above.  The new features are
          stored in the self.subsides_df data frame.
        - Bin the engineered features in self.subsidies_df by month to generate a new data frame, 
          self.binned_subsidies_df, containing the mean, median, min, max, and standard deviation 
          for each feature for each monthly bin.
    '''

    def __init__(self, username=None, crop_name='UPCN'):
        self.crop_name = crop_name
        self.futures_crop_dict = {'UPCN':'US Cotton Futures', 'CORN':'US Corn Futures', 
                         'WHEAT':'US Wheat Futures', 'SOYBEAN':'US Soybeans Futures', 
                         'RICE':'Raw Rice Futures'}
        self.weather_event_type = 'Drought'
        raw_input('Please launch Postgres, start your server, and press Enter to continue:')
        if username == None:
            self.username = raw_input('Enter your PostgreSQL username:')
        else:
            self.username = username
        
    def query_sql_database(self):
        '''
        A function to query the PostgreSQL agricultural subsidies db data base generated
        by the DataCleaner class and load the relevant data into Pandas data frames. 
        '''
        con = psycopg2.connect(database='agricultural_subsidies_db', user=self.username)
        sql_query = "SELECT * FROM subsidies_data WHERE crop_name='%s';" % self.crop_name
        self.subsidies_df = pd.read_sql_query(sql_query, con)
        self.subsidies_df = self.subsidies_df.sort_values(by='transaction_date')
        self.subsidies_df.transaction_date = pd.to_datetime(self.subsidies_df.transaction_date)
        sql_query = "SELECT * FROM weather_data;"
        self.weather_df = pd.read_sql_query(sql_query, con)
        self.weather_df = self.weather_df[self.weather_df.EVENT_TYPE == self.weather_event_type]
        sql_query = "SELECT * FROM futures_data;"
        self.futures_df = pd.read_sql_query(sql_query, con)
        self.futures_df.date = pd.to_datetime(self.futures_df.date)
        con.close()
    
    ### Features based on subsidy payment data base:
    
    def add_transaction_year(self):
        '''
        A function to parse the subsidy payment transaction date and add
        the trasaction year as a new feature.
        '''
        self.subsidies_df['transaction_year'] = \
		self.subsidies_df.transaction_date.apply(lambda x: x.year)

    def add_transaction_month(self):
        '''
        A function to parse the subsidy payment transaction date and add
        the trasaction month as a new feature.
        '''
        self.subsidies_df['transaction_month'] = \
		self.subsidies_df.transaction_date.apply(lambda x: x.month)

    def add_transaction_day(self):
        '''
        A function to parse the subsidy payment transaction date and add
        the trasaction day as a new feature.
        '''
        self.subsidies_df['transaction_day'] = \
		self.subsidies_df.transaction_date.apply(lambda x: x.day)
    
    def add_day_of_year_number(self, row):
        '''
        A function to add a new feature: number of days into the year / 365
        for each subsidy payment.
        Returns sin(2pi * days / 365) to cycle at December 31/January 1
        '''
        year = row.transaction_date.year
        days = (row.transaction_date - pd.to_datetime('%s-01-01' % year)).days
        return np.sin(2. * np.pi * days / 365.)

    def add_day_of_month_number(self, row):
        '''
        A function to add a new feature: number of days into the month / 30
        for each subsidy payment.
        Returns sin(2pi * days / 30) to cycle at first/last of month
        '''
        year = row.transaction_date.year
        month = row.transaction_date.month
        days = (row.transaction_date - pd.to_datetime('%s-%s-01' % (year, month))).days
        return np.sin(2. * np.pi * days / 30.)

    def add_season_number(self):
        '''
        A function to add a new feature: season number / 4
        for each subsidy payment.  Seasons are defined as meteorological seasons:
            - Winter = [December, January, February] = season 1
            - Spring = [March, April, May] = season 2
            - Summer = [June, July, August] = season 3
            - Autumn = [September, October, November] = season 4
        Returns sin(2pi * season / 4) to cycle yearly
        '''
        month_to_season_dict = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3,
                                7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
        self.subsidies_df['season'] = \
		self.subsidies_df.transaction_month.apply(lambda x: month_to_season_dict[x])
        self.subsidies_df.season = \
		self.subsidies_df.season.apply(lambda x: pl.cos(2. * pl.pi * (x - pl.pi/2.) / 4.))
            
    def add_week_of_year_number(self):
        '''
        A function to add a new feature: number of weeks into the year / 52
        for each subsidy payment.
        Returns sin(2pi * weeks / 52) to cycle at the start/end of the year
        '''
        self.subsidies_df['week_of_year'] = \
		self.subsidies_df.transaction_date.apply(lambda x: 
                                                         (x - pd.to_datetime('%s-01-01' % x.year)).days/7.)
        self.subsidies_df.week_of_year = \
		self.subsidies_df.week_of_year.apply(lambda x: pl.sin(2. * pl.pi * x / 52.))
    
    def add_log_transaction_amount(self):
        '''
        A function to add the natural logarithm of the subsidy payment amount as a feature.
        '''
        self.subsidies_df['log_transaction_amount'] =  \
		self.subsidies_df.transaction_amount.apply(lambda x: np.log(x))
    
    def add_customer_number_binary_var(self):
        '''
        A function to add a binary feature: 
            1 = USDA customer number starts with the letter 'A',
            0 = USDA customer number starts with any other letter (in effect, letter 'B')
        The meaning of this distinction is not documented in the data set, but may be related to
        the size or type of farm to which the payment was made (e.g., small farm vs agricultural
        conglomerate).
        '''
        self.subsidies_df['customer_number_startswith_A'] = \
		self.subsidies_df.customer_number.apply(lambda x: 1 if x.startswith('A') else 0)
            
    def add_customer_cluster_number(self):
        '''
        A function to identify clusters of customers (farms) across different states and counties.
        Returns the KMeans cluster number (from 0 to 9) for the farm, as identified by the 
        sklearn.cluster.KMeans algorithm with the number of clusters (n_clusters) set to 10.
        '''
        clustering_columns = ['log_transaction_amount', 'day_of_year', 'day_of_month', 
                              'program_year', 'program_code']
        clustering_features = self.subsidies_df[clustering_columns]
        clustering_features = sklearn.preprocessing.normalize(clustering_features)
        km = KMeans(n_clusters=10, random_state=0)
        km.fit(clustering_features)
        self.subsidies_df['cluster'] = km.labels_
        
    def add_categorical_dummy_vars(self):
        '''
        A function to generate one-hot encoded variables for the following categorical variables:
            - program_code = USDA code for the subsidy program under which a payment was made
              (e.g., program_code 2603 = USDA's "Distaster Assistance" program)
            - state_code = federal state FIPS code for the customer's state (e.g., Nebraska = 31)
            - state_county_code = federal state-and-county FIPS for the customer's state and county
              (e.g., Adams County, Nebraska = 31-001).
            - cluster = cluster number to which a customer (farm) belongs, as defined in the function
              add_customer_cluster_number.
        '''
        self.subsidies_df['state_county_code'] = self.subsidies_df.apply(lambda row: '%s_%s' % \
							  (row.state_code, row.county_code), axis=1)
        column_names = ['program_code', 'state_code', 'state_county_code', 'cluster']
        for column_name in column_names:
            dummies = pd.get_dummies(self.subsidies_df['%s' % column_name], 
                                     prefix='%s' % column_name)
            self.subsidies_sdf = pd.concat([self.subsidies_df, dummies], axis=1)  

    def run_subsidy_df_features_wrapper(self):
        '''
        A wrapper function to run each of the subsidy data feature engineering functions
        defined above.
        '''
        self.add_transaction_year()
        self.add_transaction_month() 
        self.add_transaction_day()
        self.subsidies_df['day_of_year'] = \
		self.subsidies_df.apply(self.add_day_of_year_number, axis=1)
        self.subsidies_df['day_of_month'] = \
		self.subsidies_df.apply(self.add_day_of_month_number, axis=1)
        self.add_week_of_year_number()
        self.add_season_number()
        self.add_customer_number_binary_var()
        self.add_log_transaction_amount()
        self.add_customer_cluster_number()
        self.add_categorical_dummy_vars()
    
    ### Features based on 30-day look-back window of commodities futures prices:
    
    def add_futures_1month_mean(self, row):    
        '''
        A function to add the average price of the crop-of-interest's futures price during
        a 30-day look-back window prior to the subsidy payment.
        '''
        price_df = self.futures_df[(self.futures_df.commodity==self.commodity) & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        if len(price_df.index) == 0:
            return pl.nan
        return np.mean(price_df.price)

    def add_futures_1month_std(self, row):
        '''
        A function to add the standard deviation of the price of the crop-of-interest's futures 
        price during a 30-day look-back window prior to the subsidy payment.
        '''
        price_df = self.futures_df[(self.futures_df.commodity==self.commodity) & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        if len(price_df.index) == 0:
            return pl.nan
        return np.std(price_df.price)

    def add_futures_1month_range(self, row):
        '''
        A function to add the range of the crop-of-interest's futures price during
        a 30-day look-back window prior to the subsidy payment.
        '''
        price_df = self.futures_df[(self.futures_df.commodity==self.commodity) & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        if len(price_df.price.values) == 0:
            return pl.nan
        return max(price_df.price) - min(price_df.price)
    
    def add_ethanol_futures_1month_mean(self, row):
        '''
        A function to add the average price of ethanol futures during a 30-day look-back 
        window prior to the subsidy payment.  Relevant for corn subsidies.
        '''
        price_df = self.futures_df[(self.futures_df.commodity == 'Ethanol Futures') & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        return np.mean(price_df.price)

    def add_ethanol_futures_1month_std(self, row):
        '''
        A function to add the standard deviation of the price of ethanol futures during
        a 30-day look-back window prior to the subsidy payment.  Relevant for corn subsidies.
        '''
        price_df = self.futures_df[(self.futures_df.commodity == 'Ethanol Futures') & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        return np.std(price_df.price)

    def add_ethanol_futures_1month_range(self, row):
        '''
        A function to add the range of the  price of ethanol futures during a 30-day look-back 
        window prior to the subsidy payment.  Relevant for corn subsidies.
        '''
        price_df = self.futures_df[(self.futures_df.commodity == 'Ethanol Futures') & 
                                   (self.futures_df.date < (row.transaction_date - timedelta(days=30)))]
        if len(price_df.price.values) == 0:
            return pl.nan
        return max(price_df.price) - min(price_df.price)
    
    def run_commodity_futures_features_wrapper(self):
        '''
        A wrapper function to run each of the futures data feature engineering functions
        defined above.
        '''
        self.commodity = self.futures_crop_dict[self.crop_name]
        self.subsidies_df['futures_30day_mean'] = self.subsidies_df.apply(self.add_futures_1month_mean, axis=1)
        self.subsidies_df['futures_30day_std'] = self.subsidies_df.apply(self.add_futures_1month_std, axis=1)
        self.subsidies_df['futures_30day_range'] = self.subsidies_df.apply(self.add_futures_1month_range, axis=1)
        if self.crop_name == 'CORN':
            self.subsidies_df['ethanol_futures_1month_avg'] = \
		self.subsidies_df.apply(self.add_ethanol_futures_1month_mean, axis=1)
            self.subsidies_df['ethanol_futures_1month_std'] = \
		self.subsidies_df.apply(self.add_ethanol_futures_1month_std, axis=1)
            self.subsidies_df['ethanol_futures_1month_range'] = \
		self.subsidies_df.apply(self.add_ethanol_futures_1month_range, axis=1)
    
    ## Features based on weather events
    
    def get_num_days_drought(self, row):
        '''
        A function to add the number of days of drought in the customer's county up to the
        time of the subsidy payment as a new feature for each payment record.
        '''
        year = row.transaction_year
        bool_filter = (self.weather_df.STATE_FIPS == row.state_code) & \
		      (self.weather_df.CZ_FIPS == row.county_code) & \
		      (self.weather_df.BEGIN_DATE_TIME >= pd.to_datetime('%s-01-01' % year)) & \
		      (self.weather_df.BEGIN_DATE_TIME >= row.transaction_date) 
        idxs = pl.where(bool_filter==True)[0]
        if len(idxs) == 0:
            return 0
        start_dates, end_dates = self.weather_df.BEGIN_DATE_TIME.values[idxs], \
		self.weather_df.END_DATE_TIME.values[idxs]
        durations = self.weather_df.END_DATE_TIME.values[idxs] - \
		self.weather_df.BEGIN_DATE_TIME.values[idxs]
        total_days = 0
        for i in range(len(durations)):
            duration_in_days = durations[i].astype('timedelta64[D]') / np.timedelta64(1,'D')
            total_days += duration_in_days
        return total_days
    
    def get_drought_severity_count(self, row):
        '''
        A function to assign a numeric severity score to the drought(s) that have occurred in a
        customer's county from the start of the year to the date of the subsidy payment.
          - The one-paragraph description field for each drought event is parsed and searched for
            word tokens indicative of severe droughts (e.g., word tokens corresponding to 
            "extreme," "acute", "harsh") and for word tokens indicative of mild droughts 
            (e.g., "moderate", "mild", "lessened")
          - Droughts with descriptives containing severe-dought word tokens are assigned a
            value of + 1
          - Droughts with descriptives containing mild-dought word tokens are assigned a
            value of - 1
          - Drought descriptions containing no word tokens from either the severe- or mild-drought 
            word token list, or descriptions containing tokens from both lists, are assigned a 
            value of 0.
        Returns the cumulative severity score for all drought descriptions for droughts in the customer's
        county from the start of the year to the date of the subsidy payment.
        '''
        severe_word_stems = ['extrem', 'harsh', 'harsher', 'sever', 'acut']
        mild_word_stems = ['mild', 'lessen', 'moder', 'slight']
        weather_subset = self.weather_df[(self.weather_df.EVENT_TYPE == 'Drought') & \
					 (self.weather_df.STATE_FIPS == row.state_code) & \
					 (self.weather_df.CZ_FIPS == row.county_code) & \
					 (self.weather_df.BEGIN_DATE_TIME < row.transaction_date) & \
					 (self.weather_df.END_DATE_TIME <= row.transaction_date)]
        if weather_subset.empty:
            return 0
        severity_count = 0
        narratives = weather_subset.EPISODE_NARRATIVE.values
        ps = nltk.PorterStemmer()
        for narrative in narratives:
            word_tokens = nltk.tokenize.word_tokenize(re.sub(r'[^\x00-\x7F]+','', narrative))
            word_stems = [ps.stem(w) for w in word_tokens]
            if any(stem in word_stems for stem in severe_word_stems):
                severity_count += 1
            if any(stem in word_stems for stem in mild_word_stems):
                severity_count -= 1
        return severity_count
    
    def run_weather_features_wrapper(self):
        '''
        A wrapper function to run each of the weather data feature engineering functions
        defined above.
        '''
        self.subsidies_df['days_of_drought_in_yr'] = \
		self.subsidies_df.apply(self.get_num_days_drought, axis=1)
        self.subsidies_df['drought_severity_count'] = \
		self.subsidies_df.apply(self.get_drought_severity_count, axis=1)
    
    ### Run wrapper functions to add features to subsidies data frame
    
    def engineer_new_subsidy_features(self):
        '''
        A function to call the wrapper functions defined above to engineer new
        features for each subsidy payment using data from the subsidies, weather,
        and commodity futures data sets.
        '''
        self.query_sql_database()
        self.run_subsidy_df_features_wrapper()
        if self.crop_name in self.futures_crop_dict.keys():
            self.run_commodity_futures_features_wrapper()
        self.run_weather_features_wrapper()
        
    ### Bin features of interest by month
    
    def generate_monthly_binned_df(self):
        '''
        A function to bin the engineered features in self.subsidies_df by month.  
        Produces a new data frame, self.binned_subsidies_df, containing the 
        mean, median, min, max, and standard deviation for each feature for each monthly bin.
        '''
        target_columns = ['transaction_amount']
        train_columns = ['customer_number_startswith_A', 'days_of_drought_in_yr',
                         'drought_severity_count', 'day_of_year', 'day_of_month',
                         'program_year', 'season', 'transaction_year', 'transaction_month', 
                         'transaction_day', 'week_of_year']
        if self.crop_name in self.futures_crop_dict.keys():
            train_columns += ['futures_30day_mean', 'futures_30day_std', 'futures_30day_range']
        train_columns += [col for col in self.subsidies_df.columns if col.startswith('cluster_')]
        train_columns += [col for col in self.subsidies_df.columns if col.startswith('program_code_')]
        train_columns += [col for col in self.subsidies_df.columns if col.startswith('state_code_')]
        train_columns += [col for col in self.subsidies_df.columns if col.startswith('state_county_code_')]
        if self.crop_name == 'CORN':
            train_columns += [col for col in self.subsidies_df.columns if col.startswith('ethanol_')]
        if not set(target_columns + train_columns) < set(self.subsidies_df.columns):
            raise Exception('Need to run engineer_new_subsidy_features() before binning the data by month.')

        grouped_bins = self.subsidies_df.groupby(by=['transaction_year', 'transaction_month'], 
                                                 as_index=False)
        mean_bins = grouped_bins.mean()
        median_bins = grouped_bins.median()
        std_bins = grouped_bins.std()
        rangemin_bins = grouped_bins.min()
        rangemax_bins = grouped_bins.max()

        self.binned_subsidies_df = pd.DataFrame()
        for description, bins in zip(['mean', 'median', 'std', 'rangemin', 'rangemax'], 
                                      [mean_bins, median_bins, std_bins, rangemin_bins, rangemax_bins]):
            bins['transaction_yearmonth'] = bins.apply(lambda row: '%s-%s-15' % 
                                                       (int(row.transaction_year), 
                                                        int(row.transaction_month)), axis=1)
            columns = train_columns + target_columns + ['transaction_yearmonth']
            bin_df = pd.DataFrame(columns=columns)
            for column in columns:
                bin_df[column] = bins[column]
                bin_df.transaction_yearmonth = pd.to_datetime(bin_df.transaction_yearmonth)
            bin_df.columns = [description + '_%s' % column for column in bin_df.columns]
            if description != 'mean':
                bin_df.drop('%s_transaction_yearmonth' % description, axis=1, inplace=True)
            self.binned_subsidies_df = pd.concat([self.binned_subsidies_df, bin_df], axis=1)

        self.binned_subsidies_df['log_mean_transaction_amount'] = \
		self.binned_subsidies_df.mean_transaction_amount.apply(lambda x: np.log(x))
    
    ### Run the feature engineering pipeline:
    
    def run(self):
        '''
        A function to run the entire feature engineering pipeline, as follows:
            - Load the subsidy payment, weather, and commodity futures data sets
            - For each subsidy payment record: engineer new features using the three
              data sets and the 'add_[...]' functions defined above
            - Bin the engineered features by month to generate a new data frame, 
              self.binned_subsidies_df, containing the mean, median, min, max, and 
              standard deviation for each feature for each monthly bin.
        '''
        self.engineer_new_subsidy_features()
        self.generate_monthly_binned_df()

        
if __name__ == '__main__':
    FeatureEngineer = FeatureEngineer()
    FeatureEngineer.run()