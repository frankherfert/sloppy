import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from datetime import date, datetime
import calendar

def add_datetime_features(df, datetime_columns: list, add_time_features=False, scale_0_to_1=True, 
        cos_sin_transform=True, return_new_cols=True) -> pd.DataFrame:
    """
    Create common date and time features out of given datetime-columns.
    Optional to scale between 0 and 1.
    """
    
    new_cols = []
    
    if cos_sin_transform:
        scale_0_to_1 = True
    
    # loop through all columns
    for col in datetime_columns:
        # make sure column has a datetime-
        df[col] = pd.to_datetime(df[col])
        
        df[f'ft_{col}_year']        = df[col].dt.year
        df[f'ft_{col}_month']       = df[col].dt.month        
        df[f'ft_{col}_week']        = df[col].dt.week
        df[f'ft_{col}_day']         = df[col].dt.day
        df[f'ft_{col}_dayofweek']   = df[col].dt.dayofweek
        df[f'ft_{col}_dayofyear']   = df[col].dt.dayofyear
        df[f'ft_{col}_daysinmonth'] = df[col].dt.daysinmonth
        
        # new_cols.extend([f'ft_{col}_year', f'ft_{col}_month', f'ft_{col}_week',
        #                  f'ft_{col}_day', f'ft_{col}_dayofweek',
        #                  f'ft_{col}_dayofyear', f'ft_{col}_daysinmonth'])
        
        if add_time_features:
            df[f'ft_{col}_hour'] = df[col].dt.hour
            df[f'ft_{col}_min']  = df[col].dt.minute
            df[f'ft_{col}_sec']  = df[col].dt.second
            
        if scale_0_to_1:
            max_year = df[f'ft_{col}_year'].max()
            min_year = df[f'ft_{col}_year'].min()
            df[f'ft_{col}_year']        =  (df[f'ft_{col}_year'] - min_year) / (max_year - min_year)
            
            df[f'ft_{col}_month']       = df[f'ft_{col}_month']     / 12
            df[f'ft_{col}_week']        = df[f'ft_{col}_week']      / 52
            df[f'ft_{col}_dayofweek']   = df[f'ft_{col}_dayofweek'] / 7
            df[f'ft_{col}_day']         = df[f'ft_{col}_day']       / df[f'ft_{col}_daysinmonth']
            df[f'ft_{col}_dayofyear']   = df[f'ft_{col}_dayofyear'] / 366
            df[f'ft_{col}_daysinmonth'] = df[f'ft_{col}_daysinmonth'] / 31
        
            if add_time_features:
                df[f'ft_{col}_hour'] = df[f'ft_{col}_hour'] / 24
                df[f'ft_{col}_min']  = df[f'ft_{col}_min']  / 60
                df[f'ft_{col}_sec']  = df[f'ft_{col}_sec']  / 60
                
                
        if cos_sin_transform:
            for seasonal_part in ['month', 'week', 'dayofweek']:
                df[f'ft_{col}_{seasonal_part}_cos'] = np.round(np.cos(df[f'ft_{col}_{seasonal_part}']      * 2 * np.pi), 4)
                df[f'ft_{col}_{seasonal_part}_sin'] = np.round(np.sin(df[f'ft_{col}_{seasonal_part}']      * 2 * np.pi), 4)
        
            if add_time_features:
                for seasonal_part in ['hour', 'min', 'sec']:
                    df[f'ft_{col}_{seasonal_part}_cos'] = np.round(np.cos(df[f'ft_{col}_{seasonal_part}']      * 2 * np.pi), 4)
                    df[f'ft_{col}_{seasonal_part}_sin'] = np.round(np.sin(df[f'ft_{col}_{seasonal_part}']      * 2 * np.pi), 4)
        
    new_cols = [col for col in df.columns if col.find(f'ft_{col}_')==0]
                
    if return_new_cols: return df, new_cols
    else:               return df