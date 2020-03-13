import pandas as pd
import numpy as np


def create_feature_prefix(df, columns: list, drop_orig_cols=False, return_new_col_names=True, verbose=True):
    """
    Adds a 'ft_' prefix to columns which can already be used as continous features.
    """
    new_column_names = []

    for col in columns:
        new_col_name = 'ft_'+col
        #col_name = col_name.replace('cont_cont_', 'cont_')
        df[new_col_name] = df[col]
        new_column_names.append(new_col_name)

    if return_new_col_names:
        return df, new_column_names
    else:
        return df


def create_log1p(df, columns: list, return_new_cols=True, verbose=True):
    """
    Add column with log1p conversion of values. Useful for continous values with long tail distribution.
    """
    
    new_cols = []
    
    for col in columns:
        if col.startswith('ft_'): new_column_name =         col + '__log1p'
        else:                     new_column_name = 'ft_' + col + '__log1p' 
        
        df[new_column_name] = np.log1p(df[col].clip(0,))
        
        new_cols.append(new_column_name)
            
        if verbose: print('added continous log1p column: ', 
                          '| min',  str( round(df[new_column_name].min(),2) ).rjust(5),
                          '| max',  str( round(df[new_column_name].max(),2) ).rjust(5),
                          '\t', new_column_name)
    
    return df


def create_cut_percentile(df, columns: list, percentile=0.99, return_new_cols=True, verbose=True):
    """
    Add new columns with original continous data clipped at specified percentile.
    Good for removing outlier for linear models.
    """

    new_cols = []
    
    for col in columns:
        if col.startswith('ft_'): new_column_name = f'{col}__pctl{percentile}'
        else:                     new_column_name = f'ft_{col}__pctl{percentile}'

        percentile_value = df[col].quantile(percentile)
        df[new_column_name] = df[col].clip(None, percentile_value)

        new_cols.append(new_column_name)
        if verbose:
            print('added', new_column_name, 'with upper clip at', percentile)

    if return_new_cols: return df, new_cols
    else:               return df







def cont_feature_correlation(df, columns:list):
    """
    Calculates correlation for continous features.
    """
    correlations = df.loc[:, columns].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    
    correlations.columns = ['feat_1', 'feat_2', 'correlation']

    correlations['features'] = correlations['feat_1'] + '<>' + correlations['feat_2']
    correlations['features'] = correlations['features'].apply(lambda s: ' <> '.join( sorted(s.split('<>'))))
    
    correlations = correlations.sort_values(['feat_1', 'feat_2'])
    correlations = correlations.drop_duplicates('features')
    correlations = correlations.loc[:, ['feat_1', 'feat_2', 'features', 'correlation']]
    return correlations
