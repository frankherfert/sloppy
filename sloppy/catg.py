import pandas as pd
import numpy as np
import random
import warnings
from sklearn import preprocessing


def convert_to_pd_catg(df, columns: list, verbose=True) -> pd.DataFrame:
    """
    Converts all columns to pandas categorical type.
    Enables additional functions and more memory-efficient data handling.
    """
    for col in columns:
        try:
            df[col]  = df[col].astype('category')
            if verbose: print('converted to categorical:', col)
        except:
            print('error for               :', col)

    return df


def add_count_encoding(df, column_combinations: list, scaler: 'sklearn.preprocessing. ...' = None,
                       verbose = True, drop_orig_cols=False):
    """
    Expects a DataFrame with no missing values in specified columns.
    Creates new columns for every column combination (one or more columns to be combined).

    :df:                    DataFrame
    :column_combinations:   list of single or multiple columns,
                            eg.: ['country', 'product', ['country', 'product']]
    :scaler:                sklearn sccaler for normalization
    :drop_orig_cols:        drop original columns after count-encoding
    """

    # create temp-column with no missing values, used for counting
    df['tmp'] = 1
    
    for cols in column_combinations:
        # set name suffix for new column
        if isinstance(cols, list):
            var_name = '_'.join(cols)
        else:
            var_name = cols
        
        new_column_name = 'cont_' + var_name + '__count' 
        
        # groupby count transform
        counts = df.groupby(cols)['tmp'].transform('count').values.reshape(-1, 1)#.astype(int)
        
        if scaler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     

                counts = scaler.fit_transform(counts);
                scaler_str = str(type(scaler)).split('.')[-1].split('Scaler')[0].split('Transformer')[0].lower()
                new_column_name = f'{new_column_name}_scaled_{scaler_str}'
            
        df[new_column_name] = counts

        if verbose: print('added categorical column count: ', 
                          'unique', str( df[new_column_name].nunique() ).rjust(5), 
                          '| min',  str( df[new_column_name].min()     ).rjust(5),
                          '| max',  str( df[new_column_name].max()     ).rjust(5),
                         '\t', new_column_name)
        
        if drop_orig_cols:
            df = df.drop(cols, axis=1)
    
    df = df.drop('tmp', axis=1)
    return df


def add_label_encoding(df, column_combinations: list, min_count = 5, 
                       drop_orig_cols = False, verbose = True):
    """
    Add numerical labels for categorical values.
    Values under a specified low total count are grouped together as '0'
    """
    #max_col_length = len(max(columns, key=len))

    # set name suffix for new column
    for col_combo in column_combinations:
        var_name = '_'.join(col) if isinstance(col, list) else col
        new_column_name = 'catg_' + var_name + '__label'
        
        # if label encoding for column combination, use groupby.transform to get unique values per combination
        # if single column, use values directly
        if isinstance(col_combo, list):
            column_values = df.groupby(col_combo)[col_combo[0]].transform(lambda series: random.random)
        else:
            column_values = df[col].copy()

        # determine low-count outliers, replace with '00000'
        col_counts = df.groupby(col)[col].transform("count")
        column_values = np.where(col_counts < min_count, '00000', column_values)

        # label encode remaining values
        label_encoder = preprocessing.LabelEncoder()
        df[new_column_name] = label_encoder.fit_transform(column_values)

        if verbose:
            print(new_column_name.ljust(20),
                  "unique:", str(len(df[col].unique())).ljust(8),
                  "\t ids:", len(df[new_column_name].unique()))
        
        if drop_orig_cols:
            df = df.drop(cols, axis=1)
    
    return df


def add_one_hot_encoding(df, columns: list, min_pctg_to_keep=0.03, verbose=True):
    """
    Adds one-hot encoded columns for each categorical column
    """
    max_col_length = len(max(columns, key=len))
    
    for column in columns:
        #df[column]    = df[column].apply(lambda x: str(x)) #convert to str just in case
        #new_columns = [column + "_" + i for i in full[column].unique()] #only use the columns that appear in the test set and add prefix like in get_dummies
        
        one_hot_df = pd.get_dummies(df[column], prefix=f'cont_{column}__one_hot_')
        orig_col_number = len(one_hot_df.columns)
        
        keep = (one_hot_df.sum()/len(one_hot_df))>=min_pctg_to_keep
        one_hot_df = one_hot_df.loc[:, keep]

        if verbose:
            print('created one-hot-encodings:', column.ljust(max_col_length),
             f'  -  keep {len(one_hot_df.columns)}/{orig_col_number} one-hot columns')
   
        df = pd.concat((df, one_hot_df), axis = 1)
   
    return df