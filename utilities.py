#import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
sns.set_context(context='paper',font_scale=1.5)

from scipy.signal import savgol_filter as smooth
import scipy.stats as stats

#import metrics
from sklearn.metrics import r2_score, mean_squared_error, \
                            mean_absolute_error, max_error

#import linear regression model
from sklearn.linear_model import LinearRegression

#add to make folders when needed
import os


#################
### Functions ###
#################

def mkfolder(folder):
    #make the directory
    try:
        os.mkdir(folder)
    except:
        print(f'Folder {folder} already exists.')
        
#least square fit to level the data
def ols(series):
    """This function takes a pandas series and returns coefficients 
    from linear regression, assuming step = 1."""
    
    ##fit linear relationship
    lin_model = LinearRegression()
    lin_model.fit(np.arange(len(series)).reshape(-1, 1),series)
    
    #slope
    m = lin_model.coef_[0]
    #intercept
    b = lin_model.intercept_

    return m, b

#function to read data in sliding window format
def sliding_windows(ts, n_inputs, n_outputs, shift = 0):
    """This function takes a time series (ts) and slices it
    according number of inputs (n_inputs) for x;
    number of outputs (n_outputs) and shift points (shift) for y.
    Useful for cross-validation with sliding window."""
    
    x = []
    y = []

    for i in range(0,len(ts) - n_inputs - n_outputs, n_outputs):
        _x = ts.iloc[i-shift:(i+n_inputs)-shift]
        _y = ts.iloc[i+n_inputs-shift:i+n_inputs+n_outputs-shift]
        x.append(_x)
        y.append(_y)

    return x, y

def get_slope(y):
    """This function computes slope estimate."""
    
    x = range(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

def get_acceleration(y):
    """This function computes acceleration estimate."""
    
    x = range(len(y))
    acc = 0.5*np.polyfit(x, y, 2)[0]
    return acc

#######################################################
################ data transformation ##################
#######################################################

def read_data(csv_file, window):
    """This function reads price data
    downloaded from datahub.io
    adds log-transformed and leveled column
    as well as smoothed values with window = 2*window + 1"""
    
    #load weekly data
    df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
    df.reset_index(inplace=True)
    
    #time range:
    print('Time range: from', df.Date.min(),'to', df.Date.max())
    #df.head()

    #log transform and level the data
    
    #log transform
    ref = df.Price.iloc[0]
    df['log_price'] = np.log(df.Price/ref)

    #remove trend by fitting 
    m, b = ols(df['log_price'])
    
    #leveled_log_price
    df['y'] = df['log_price'] - (b + df.index * m)
    
    #add column with smoothed original price data
    df[f'y_smooth_w{window}'] = smooth(df['y'].values, 2*window+1, 3)
    
    df.drop(columns = ['log_price'], inplace = True)
   
    return df, m, b, ref

def inverse_transform(data, m, b, ref, start_index = 0):
    """This function takes inverse transform of 
    leveled and log transformed data."""
    #m = 0.00022133350755924386, b =-0.6123805778581584, ref = 25.56
    
    index = np.arange(start_index, len(data) + start_index).reshape(-1,1)
    return np.exp(np.array(data).reshape(-1,1) + (b + index * m)) * ref


#######################################################
################### model metrics #####################
#######################################################

def mape(y_true, y_pred): 
    """This function computes mean average percentage error (MAPE)."""
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(y_true, y_pred):
    """This function takes true and predicted values
    and prints metrics """
    
    print('##########################')
    print('Evaluate model performance')
    print('##########################')
    print('Mean Absolute Percentage Error (MAPE):')
    print(mape(y_true, y_pred))
    print('Mean Absolute Error (MAE):')
    print(mean_absolute_error(y_true, y_pred))
    print('Root Mean Square Error (RMSE):')
    print(np.sqrt(mean_squared_error(y_true, y_pred)))
    print('Max Error (ME):')
    print(max_error(y_true, y_pred))
    print('R2 score:')
    print(r2_score(y_true, y_pred))
    

def metric_df(y_true, y_pred):
    """This function takes true and predicted values
    and writes metrics in pandas dataframe format."""
    
    return pd.DataFrame({'MAPE': mape(y_true, y_pred),
                         'MAE': mean_absolute_error(y_true, y_pred),
                         'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                         'ME': max_error(y_true, y_pred),
                         'R2 score': r2_score(y_true, y_pred)}, index = [0])

def metric_df_size(y_true, y_pred, _input, _shift, loss):
    """This function takes true and predicted values,
    number of input points and shift
    and writes metrics in pandas dataframe format."""
    
    return pd.DataFrame({'Input': _input,
                         'Shift': _shift,
                         'MAPE': mape(y_true, y_pred),
                         'MAE': mean_absolute_error(y_true, y_pred),
                         'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                         'ME': max_error(y_true, y_pred),
                         'R2 score': r2_score(y_true, y_pred),
                         'Loss': loss}, index = [0])

def metric_summary(df, column, alpha, folder, model_name, figname, figsize=(7,3)):
    """This function taked df with metrics
    computed for diffetent train-test splits
    and it computes mean value with confidence interval
    and plots the distribution,
    finally saves it in folder."""
    #compute 68% confidence interval for MAPE 
    _mean = df[column].mean()
    _std = df[column].std()
    conf_int = stats.t.interval(alpha = alpha,    # Confidence level
                    df = len(df)-1,            # Degrees of freedom
                    loc = _mean,      # Sample mean
                    scale = _std)     # Standard deviation estimate

    delta = conf_int[1] - _mean

    #plot Mean Average Percentage Error distribution for cv_num train-test splits
    fig, ax = plt.subplots(1,1,figsize=figsize)
    df[column].hist(bins=20)
    ax.set_title(f'Distribution of {figname} for {model_name} model: ' + 
                f'{column} = {round(_mean,2)} +/- {round(delta,2)}')
    ax.set_ylabel('Count')
    ax.set_xlabel(f'{column}')

#     _,x2 = ax.get_xlim()
#     _,y2 = ax.get_ylim()
#     ax.text(0.3*x2,0.6*y2,
#             f'{column} = {round(_mean,1)} +/- {round(delta,1)}',
#             fontsize=14).set_bbox(dict(edgecolor='k',facecolor='w'))

    plt.tight_layout()
    #save figure in the folder
    fig.savefig(f'{folder}/{figname}_hist.png')