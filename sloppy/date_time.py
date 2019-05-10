import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from datetime import date, datetime
import calendar

def add_datetime_features(df, datetime_columns: list, add_time_features=False, scale_0_to_1=True, cos_sin_transform=True) -> pd.DataFrame:
    """
    put text here
    """
    
    if cos_sin_transform:
        scale_0_to_1 = True
    
    # loop through all columns
    for col in datetime_columns:
        # make sure column has a datetime-
        df[col] = pd.to_datetime(df[col])
        
        df[f'dt_{col}_year']        = df[col].dt.year
        df[f'dt_{col}_month']       = df[col].dt.month        
        df[f'dt_{col}_week']        = df[col].dt.week
        df[f'dt_{col}_day']         = df[col].dt.day
        df[f'dt_{col}_dayofweek']   = df[col].dt.dayofweek
        df[f'dt_{col}_dayofyear']   = df[col].dt.dayofyear
        df[f'dt_{col}_daysinmonth'] = df[col].dt.daysinmonth
        
        if add_time_features:
            df[f'dt_{col}_hour'] = df[col].dt.hour
            df[f'dt_{col}_min']  = df[col].dt.minute
            df[f'dt_{col}_sec']  = df[col].dt.second
            
        if scale_0_to_1:
            max_year = df[f'dt_{col}_year'].max()
            min_year = df[f'dt_{col}_year'].min()
            df[f'dt_{col}_year']        =  (df[f'dt_{col}_year'] - min_year) / (max_year - min_year)
            
            df[f'dt_{col}_month']       = df[f'dt_{col}_month']     / 12
            df[f'dt_{col}_week']        = df[f'dt_{col}_week']      / 52
            df[f'dt_{col}_dayofweek']   = df[f'dt_{col}_dayofweek'] / 7
            df[f'dt_{col}_day']         = df[f'dt_{col}_day']       / df[f'dt_{col}_daysinmonth']
            df[f'dt_{col}_dayofyear']   = df[f'dt_{col}_dayofyear'] / 366
            df[f'dt_{col}_daysinmonth'] = df[f'dt_{col}_daysinmonth'] / 31
        
            if add_time_features:
                df[f'dt_{col}_hour'] = df[f'dt_{col}_hour'] / 24
                df[f'dt_{col}_min']  = df[f'dt_{col}_min']  / 60
                df[f'dt_{col}_sec']  = df[f'dt_{col}_sec']  / 60
                
                
        if cos_sin_transform:
            for seasonal_part in ['month', 'week', 'dayofweek']:
                df[f'dt_{col}_{seasonal_part}_cos'] = np.round(np.cos(df[f'dt_{col}_{seasonal_part}']      * 2 * np.pi), 4)
                df[f'dt_{col}_{seasonal_part}_sin'] = np.round(np.sin(df[f'dt_{col}_{seasonal_part}']      * 2 * np.pi), 4)
        
            if add_time_features:
                for seasonal_part in ['hour', 'min', 'sec']:
                    df[f'dt_{col}_{seasonal_part}_cos'] = np.round(np.cos(df[f'dt_{col}_{seasonal_part}']      * 2 * np.pi), 4)
                    df[f'dt_{col}_{seasonal_part}_sin'] = np.round(np.sin(df[f'dt_{col}_{seasonal_part}']      * 2 * np.pi), 4)
                
        # sorted(dt_df.columns, reverse=True)
    return df