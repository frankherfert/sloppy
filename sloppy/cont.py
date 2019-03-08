import pandas as pd
import numpy as np


def add_log1p(df, columns: list, verbose=True):
    """
    """
    
    for col in columns:
        new_column_name = 'cont_' + col + '__log1p' 
        
        df[new_column_name] = np.log1p(df[col])
        
        if verbose: print('added continous log1p column: ', 
                          '| min',  str( df[new_column_name].min() ).rjust(5),
                          '| max',  str( df[new_column_name].max() ).rjust(5),
                         '\t', new_column_name)
    
    return df


def add_cut_percentile(df, columns: list, percentile=0.99, verbose=True):
    """
    Add new columns with original continous data clipped at specified percentile.
    Good for removing outlier for linear models.
    """

    for col in columns:
        new_column_name = f'cont_{col}__pctl{percentile}'

        percentile_value = df[col].quantile(percentile)
        df[new_column_name] = df[col].clip(0, percentile_value)

    if verbose:
        print('added', new_column_name, 'with upper clip at', percentile)

    return df


def round_to_nearest_int(value, base=25):
    """
    Rounds a number to the nearest base as integer value
    """
    try:
        return int(base * round(float(value)/base))
    except:
        return np.nan





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
