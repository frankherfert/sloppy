import pandas as pd
import numpy as np

def show_missing(df, columns:list = None):
    """
    Shows number and percentage of rows with missing values.
    """
    total = df.isnull().sum()
    percent = np.round((df.isnull().sum()/df.isnull().count()*100), 4)
    
    tt = pd.concat([total, percent], axis=1, keys=['missing', 'percent'])
    types = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return tt


def add_isnull_columns(df, columns, min_percentage=0.01, verbose=True):
    """
    Adds new columns with 0/1 flags for missing values.
    :min_percentage: Minimum percentage of missing values in a column needed to create a new '_isnull' column
    """
    max_col_length = len(max(columns, key=len))
    
    for column in columns:
        null_count = df[column].isnull().sum()
        null_pctg  = null_count / len(df)
        
        null_count_str = '{:,}'.format(null_count)
        null_pctg_str  = '{:.2f}%'.format(null_pctg*100)
        
        if verbose:
            print(f"add '_isnull' column for: {column.ljust(max_col_length)} \t missing: {str(null_count_str).rjust(7)} ({str(null_pctg_str).rjust(6)})")

        if null_pctg > min_percentage:
            col_name = f'cont_{column}__isnull'
            df[col_name] = np.where(df[column].isnull(), 1, 0)

    return df


def fillna_categorical(df, columns:list, fill_value:str = 'MISSING', verbose=True):
    """
    Fillas categorical columns with values
    """
    for column in columns:
        df[column] = df[column].fillna(fill_value)
        if verbose: print(f'fillna with \'{fill_value}\': {column}')

    return df


def fillna_cont_static(df, columns: list, fill_value = -1, drop_orig_cols=False, verbose=True):
    """
    fillna continous columns with a static value.
    """
    max_col_length = len(max(columns, key=len))
    
    if fill_value<0:
        fill_value_str = f'fillna_neg{abs(fill_value)}'.replace('.', '_')
    else:
        fill_value_str = f'fillna_{fill_value}'

    for column in columns:
        new_column_name = f'{column}__{fill_value_str}'
        try:
            df[new_column_name] = df[column].fillna(fill_value)
            if verbose: print(f'fillna with \'{fill_value}\': {column.ljust(max_col_length)} -> {new_column_name}')
            if drop_orig_cols: 
                df = df.drop(column, axis=1)
        except:
            print('Error for:', column)
    
    return df


def fillna_cont_groupby_value(df, columns: list, catg_groupby_columns: list, function = np.mean, drop_orig_cols=False, verbose = True):
    """
    fillna continous columns with groupby-values from a specified function.
    :catg_groupby_columns: categorical columns to use in groupby aggregation
    :function:             np.mean, np.median, ...
    """
    
    for column in columns:
        try:
            function_name = str(function).split(' ')[1]
            new_column_name = f'{column}__fillna_{function_name}_groupby_'+'_'.join(catg_groupby_columns)

            fill_values = df.groupby(catg_groupby_columns)[column].transform(function)

            df[new_column_name] = df[column].fillna(fill_values)
            if verbose: print(f'fillna with {str(function).split(" ")[1]}: {column} -> {new_column_name}')

            if drop_orig_cols: 
                  df = df.drop(column, axis=1)
        except:
            print('error for', column)

    return df
