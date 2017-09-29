import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import state_abbreviations
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

class DataCleaner:
    '''
    The DataCleaner class creates a PostgreSQL database from public data stored in .csv files
    in your local /data folder.  The .csv files were collected from the following sources:
        - Farm Services Agency subsidy data (2012 - 2016): data culled by Enigma Public from 
          Freedom of Information Act (FOIA) requests to the US Department of Agriculture. 
          https://public.enigma.com/browse/u-s-farm-subsidies/22fc74ac-f530-4c65-bd72-cebebf426144
        - Weather data (2012 - 2016) from the National Oceanic and Atmospheric Administration (NOAA)
          Storm Events DatabaseL https://www.ncdc.noaa.gov/stormevents/ftp.jsp
        - Agricultural commodities futures prices (2012 - 2016) for the "Big Five" US subsidized crops, 
          from Investing.com (CME or CBOT ticker symbol):
            - Ethanol (1ZEc1) (relevant to corn): https://www.investing.com/commodities/ethanol-futures-historical-data
            - US Corn (ZCZ7): https://www.investing.com/commodities/us-corn
            - US Cotton #2 (CTZ7): https://www.investing.com/commodities/us-cotton-no.2-historical-data
            - US Wheat (ZWZ7): https://www.investing.com/commodities/us-wheat
            - Rough Rice (RRX7): https://www.investing.com/commodities/rough-rice-historical-data
            - Soybeans (ZSX7): https://www.investing.com/commodities/us-soybeans-historical-data
          These data files are only available upon signing up for a free account with Investing.com, and
          therefore have not been posted to this repository.
    '''

    def __init__(self):
        self.raw_weather_data_files = [f for f in os.listdir('./../data') if f.startswith('StormEvents_details')]
        self.raw_subsidies_data_files = [f for f in os.listdir('./../data') if 
                                         f.startswith('USFarmSubsidiesProducerPayment')]
        self.state_code_dict, self.county_code_dict = self.generate_state_and_county_code_dicts()

    #### Weather data frame ####

    def load_raw_weather_df(self):
        for i in range(len(self.raw_weather_data_files)):
            if i == 0:
                weather_df = pd.read_csv(os.path.join('./../data', self.raw_weather_data_files[0]))
            else:
                temp_df = pd.read_csv(os.path.join('./../data', self.raw_weather_data_files[i]))
                weather_df = weather_df.append(temp_df, ignore_index=True)
        del temp_df
        return weather_df

    def damages_to_numeric(self, dmg_string):
        if type(dmg_string) == float:
            return dmg_string
        elif dmg_string.endswith('K'):
            return 1e3 * float(dmg_string.replace('K', ''))
        elif dmg_string.endswith('M'):
            return 1e6 * float(dmg_string.replace('M', ''))
        elif dmg_string.endswith('B'):
            return 1e9 * float(dmg_string.replace('B', ''))
        return pl.nan
    
    def parse_weather_dates(self): 
        weather_df['BEGIN_YEAR'] = weather_df.BEGIN_DATE_TIME.apply(lambda x: x.year)
        weather_df['BEGIN_MONTH'] = weather_df.BEGIN_DATE_TIME.apply(lambda x: x.month)
        weather_df['BEGIN_HOUR'] = weather_df.BEGIN_DATE_TIME.apply(lambda x: x.hour)
        weather_df['BEGIN_MINUTE'] = weather_df.BEGIN_DATE_TIME.apply(lambda x: x.minute)    
        weather_df['END_YEAR'] = weather_df.END_DATE_TIME.apply(lambda x: x.year)
        weather_df['END_MONTH'] = weather_df.END_DATE_TIME.apply(lambda x: x.month)
        weather_df['END_HOUR'] = weather_df.END_DATE_TIME.apply(lambda x: x.hour)
        weather_df['END_MINUTE'] = weather_df.END_DATE_TIME.apply(lambda x: x.minute)
        weather_df.drop(['BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'END_YEARMONTH', 
                         'END_DAY', 'END_TIME', 'YEAR', 'MONTH_NAME'], axis=1, inplace=True)

    def get_cleaned_weather_data(self):
        if 'processed_weather_data.csv' not in os.listdir('./../data'):
            weather_df = self.load_raw_weather_df()
            weather_df.BEGIN_DATE_TIME = pd.to_datetime(weather_df.BEGIN_DATE_TIME)
            weather_df.END_DATE_TIME = pd.to_datetime(weather_df.END_DATE_TIME)
            self.parse_weather_dates()
            weather_df.DAMAGE_CROPS = weather_df.DAMAGE_CROPS.apply(lambda x: self.damages_to_numeric(x))
            weather_df.DAMAGE_PROPERTY = weather_df.DAMAGE_PROPERTY.apply(lambda x: self.damages_to_numeric(x))
            weather_df.to_csv('./../data/processed_weather_data.csv', index=False)
        else:    
            weather_df = pd.read_csv('./../data/processed_weather_data.csv', header=0)
            weather_df.BEGIN_DATE_TIME = pd.to_datetime(weather_df.BEGIN_DATE_TIME)
            weather_df.END_DATE_TIME = pd.to_datetime(weather_df.END_DATE_TIME)
        return weather_df


    #### Subsidies Data ####

    def load_raw_subsidies_df(self):
        for i in range(len(self.raw_subsidies_data_files)):
            if i == 0:
                subsidies_df = pd.read_csv(os.path.join('./../data', raw_subsidies_data_files[0]))
            else:
                temp_df = pd.read_csv(os.path.join('./../data', raw_subsidies_data_files[i]))
                subsidies_df = subsidies_df.append(temp_df, ignore_index=True)
        del temp_df
        return subsidies

    def get_state_county_df(self):
        state_county_df = pd.read_csv('US_county_FIPS.csv')
        state_county_df.columns = ['state','county','state_code','county_code']
        state_county_df.county = state_county_df.county.apply(lambda x: x+' County')
        return state_county_df

    def generate_state_and_county_code_dicts(self):
        state_county_df = self.get_state_county_df()
        state_code_dict = {}    # dict[state_code] = state
        county_code_dict = {} # dict[state_code][county_code] = county
        for idx in state_county_df.index:
            row = state_county_df.iloc[idx]
            state_code = row.state_code
            state = row.state
            if state_code not in county_code_dict.keys():
                county_code_dict[state_code] = {}
            county_code = row.county_code
            county = row.county
            county_code_dict[state_code][county_code] = county
            if state_code in state_code_dict.keys():
                continue
            else:
                state_code_dict[state_code] = state
        return state_code_dict, county_code_dict

    def get_state_name_from_row(self, row):
        try:
            state = self.state_code_dict[row.state_code]
        except:
            raise Exception('Error extracting state name from subsidy data set row: \n %s' % row)
        return state

    def get_county_name_from_row(self, row):
        try:
            county = self.county_code_dict[row.state_code][row.county_code]
        except:
            county = pl.nan
        return county

    def get_state_abbreviation_dict(self):
        state_abbrev_dict = {}
        for state in us_state_abbrev.keys():
            abbreviation = us_state_abbrev[state]
            state_abbrev_dict[abbreviation] = state
        return state_abbrev_dict

    def genertate_commodity_code_dict(self):
        commodity_codes_df = pd.read_csv('./../data/USFarmSubsidiesMetadata-Category-Program-CommodityCodes.csv')
        commodity_dict = {}
        for code in np.sort(commodity_codes_df.commodity_code.unique()):
            names = commodity_codes_df[commodity_codes_df.commodity_code==code].commodity_name.unique()
            if len(names) > 0:
                print code, names
                name = names[0]
                if type(name) == float and pl.isnan(name):
                    name = 'UNSPECIFIED_NAN'
                commodity_dict[code] = name
            else:
                commodity_dict[pl.nan] = 'UNSPECIFIED_COMMODITY_CODE_IS_NAN'
        return commodity_dict

    def get_commodity_name_from_row(row):
        try:
            crop_name = commodity_dict[row.commodity_code]
        except:
            crop_name = None
        return crop_name

    def get_cleaned_subsidies_data(self):
        if 'processed_subsidies.csv' not in os.listdir('./../data'):
            subsidies_df = self.load_raw_subsidies_df()
            subsidies_df = subsidies_df[subsidies.state_code <= 50] # ignore data from US territories
            subsidies_df['state'] = subsidies_df.apply(self.get_state_name_from_row, axis=1)
            subsidies_df['county'] = subsidies_df.apply(self.get_county_name_from_row, axis=1)
            state_abbreviation_dict = self.get_state_abbreviation_dict()
            subsidies_df.state = subsidies_df.state.apply(lambda x: state_abbreviation_dict[x])
            subsidies_df['crop_name'] = subsidies_df.apply(self.get_commodity_name_from_row, axis=1)
            subsidies_df.to_csv('./../data/processed_subsidies.csv', index=False)
        else:
            subsidies_df = pd.read_csv('./../data/processed_subsidies.csv', header=0)
        return subsidies_df

    ## Commodity Futures Price Data

    def load_raw_futures_data(self):
        crops = ['Corn', 'Corn', 'Cotton', 'Wheat', 'Rice', 'Soybeans']
        commodieis = ['Ethanol Futures', 'US Corn Futures', 'US Cotton Futures', 'US Wheat Futures', 
                      'Raw Rice Futures', 'US Soybeans Futures']
        raw_commodity_files = ['%s_futures_prices.csv' % commodity for commodity 
                               in ['ethanol', 'corn', 'cotton', 'wheat', 'rice', 'soybeans']]

        futures_df = pd.DataFrame(columns=e['date', 'price', 'open', 'high', 'low', 'volume', 'pct_change'])
        for crop, commodity, commodity_file in zip(crops, commodities, raw_commodities_files):
            df = pd.read_csv(os.path.join('./../data', commodity_file), nrows=1260, header=0,
                              names=futures_df.columns)
            df['commodity'] = commodity
            df['relevant_crop'] = crop
            futures_df = futures_df.append(df)
        return futures_df


    def get_cleaned_futures_data(self):
        if 'processed_futures_prices.csv' not in os.listdir('./../data'):
            notice = 'NOTE: you will need to download the historical price data from investing.com and place'
            notice += ' them in your /data folder (see the README.md file for details.  Press ENTER to continue...'
            raw_input(notice)
            
            futures_df = self.load_raw_futures_data()
            futures_df.date = pd.to_datetime(futures_df.date)
            for column in ['price','open','high','low']:
                futures_df[column] = futures_df[column].apply(lambda x: x.replace(',','') if type(x)==str else x)
                futures_df[column] = futures_df[column].apply(lambda x: pd.to_numeric(x))
            futures_df.volume = futures_df.volume.apply(lambda x: 1000.*float(x.replace('K','')) if \
                                                        'K' in x else float(x))
            futures_df.to_csv('./../data/processed_futures_prices.csv', index=False)
        else:    
            futures_df = pd.read_csv('./../data/processed_futures_prices.csv', header=0)
        return futures_df

    #### Run and write data to PostgreSQL database #####

    def write_PostgreSQL_db(self, weather_df, subsidies_df, futures_df):
        raw_input('Please launch Postgres, start your server, and press Enter to continue . . .') 
        dbname = 'agricultural_subsidies_db'
        username = raw_input('Enter your PostgreSQL username:')
        engine = create_engine('postgres://%s@localhost/%s' % (username, dbname))
        print 'Creating Postgres database %s' % engine.url
        if not database_exists(engine.url):
            create_database(engine.url)
        
        print 'Writing weather data table . . .'
        weather_df.to_sql('weather_data', engine, if_exists='replace')
        print 'Writing subsidies data table . . .'
        subsidies_df.to_sql('subsidies_data', engine, if_exists='replace')
        print 'Writing commodities futures data table . . .'
        futures_df.to_sql('futures_data', engine, if_exists='replace')
        print 'Finished'

    def run(self):
        weather_df = self.get_cleaned_weather_data()
        subsidies_df = self.get_cleaned_subsidies_data()
        futures_df = self.get_cleaned_futures_data()
        self.write_PostgreSQL_db(weather_df, subsidies_df, futures_df)

        
if __name__ == '__main__':
    DataCleaner = DataCleaner()
    DataCleaner.run()
