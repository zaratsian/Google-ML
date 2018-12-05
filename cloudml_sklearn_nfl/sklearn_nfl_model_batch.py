

########################################################################################################
#
#   Sklean - Batch Scoring example
#
#   Usage:
#       python sklearn_nfl_model_batch.py --path_to_data="nfldata2.csv" --path_to_model="/tmp/model.joblib"
#
########################################################################################################

import os,sys,csv,re
import time,datetime
import argparse

import pandas as pd
from tabulate import tabulate

from sklearn.externals import joblib

########################################################################################################
#
#   Load Data
#
########################################################################################################

def load_rawdata(training_data):
    '''
        NOTE:   This function will look for a CSV file that is located in the ../data directory.
        
        USAGE:  rawdata = load_rawdata(training_data='nfldata2.csv')
    '''
    try:
        training_data = os.path.join( os.getcwd().replace('/sklearn_nfl','/data') , training_data)
        print('[ INFO ] Reading in training data from from {}'.format(training_data))
        time.sleep(3)
        
        header  = ['Date', 'GameID', 'Drive', 'qtr', 'down', 'time', 'TimeUnder', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'FirstDown', 'posteam', 'DefensiveTeam', 'Yards_Gained', 'Touchdown', 'PlayType', 'PassLength', 'PassLocation', 'RunLocation', 'PosTeamScore', 'DefTeamScore', 'month_day', 'PlayType_lag']
        rawdata = pd.read_csv(training_data , names=header)
        # Quickly explore data structure
        rawdata.head()
        #rawdata.values
        rawdata.iloc[0]
        rawdata.shape
        rawdata.columns
        rawdata.dtypes
        rawdata.describe()
        print('[ INFO ] Read in training_data located at {}'.format(training_data))
        return rawdata
    except:
        print('[ ERROR ] Could not find training_data. Check directory path and filename, then try again.')
        sys.exit()


########################################################################################################
#
#   Transformations
#
########################################################################################################

def transform_df(rawdata, target_variable_name=None):
    
    # Model Variables (Specify id, target, numeric variables, and categorical variables)
    var_id              = ''
    var_target          = target_variable_name #'Yards_Gained'
    var_date            = 'Date'
    var_numeric         = ['Drive', 'qtr', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'PosTeamScore', 'DefTeamScore', 'FirstDown'] #, 'month_day']
    var_category        = ['posteam', 'DefensiveTeam','PlayType','PlayType_lag']
    
    transformed_set             = {}
    if var_target != None:
        transformed_set[var_target] = rawdata[var_target]
    
    for var in var_numeric:
        transformed_set[var] = rawdata[var]
    
    '''
    for var in var_category:
        transformed_set[var] = rawdata[var].astype('category').cat.codes
    '''
    
    category_coding = {}
    for var in var_category:
        category_coding[var] = dict( enumerate( rawdata[var].astype('category').cat.categories ))
        transformed_set[var] = rawdata[var].astype('category').cat.codes
    
    extracted_year  = pd.to_datetime(rawdata[var_date]).dt.year
    extracted_month = pd.to_datetime(rawdata[var_date]).dt.month
    extracted_day   = pd.to_datetime(rawdata[var_date]).dt.day
    
    transformed_set['year']  = extracted_year
    transformed_set['month'] = extracted_month
    transformed_set['day']   = extracted_day
    
    # Create transformed DF
    transformed_df = pd.concat([v for k,v in transformed_set.items()], axis=1)
    transformed_df.columns = [k for k,v in transformed_set.items()]
    transformed_df.head()
    return transformed_df


########################################################################################################
#
#   Score data using Trained / Saved Model
#
########################################################################################################

def score_data(path_to_model, df_to_score):
    
    print('[ INFO ] Loading model from {}'.format(path_to_model))
    model_obj = joblib.load(path_to_model)
    
    scores = model_obj.predict(df_to_score)
    df_to_score['predicted'] = scores
    
    # Returns dataframe with predictions appended to last column
    return df_to_score


def save_scored_df(scored_df):
    try:
        scored_data_path = '/tmp/scored_nfl_data.csv'
        scored_df.to_csv(scored_data_path)
        print('[ INFO ] Scored dataframe successfully save to /tmp/{}'.format(scored_data_path))
    except:
        print('[ WARNING ] Could not save scored data')



if __name__ == "__main__":
    
    # Arguments (used only for testing)
    #args = {"path_to_data":"nfldata2.csv", "path_to_model":"/tmp/model.joblib"}
    
    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_to_data",   required=True,  type=str,   help="Path to data")
    ap.add_argument("--path_to_model",  required=True,  type=str,   help="Path to model (.joblib)")
    args = vars(ap.parse_args())
    
    # Load Dataset
    rawdata = load_rawdata(args['path_to_data'])
    
    # Transform / Prep dataframe
    transformed_df = transform_df(rawdata, None)
    
    # Score Data
    scored_df = score_data(args['path_to_model'], transformed_df)
    scored_df['actual'] = rawdata['Yards_Gained']
    
    # Print the Results
    print(tabulate(scored_df[:50].filter(['Drive','qtr','down','TimeSecs','ydstogo','predicted','actual']), headers='keys', tablefmt='psql'))
    
    # Save Scored DF to local
    save_scored_df(scored_df)



#ZEND
