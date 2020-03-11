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
    if verbose: print('converting to categorical...')
    for col in columns:
        try:
            if verbose: print(f'- {col}', end=' ')
            df[col]  = df[col].astype('category')
            if verbose: print('ok')
        except:
            print(' error')

    return df


def create_count_encoding(df, column_combinations: list, scaler: 'sklearn.preprocessing. ...' = None,
                       verbose = True, return_new_cols = True,
                       drop_orig_cols=False) -> pd.DataFrame:
    """
    Expects a DataFrame with no missing values in specified columns.
    Creates new columns for every column combination (one or more columns to be combined).

    :df:                    DataFrame
    :column_combinations:   list of single or multiple columns,
                            eg.: ['country', 'product', ['country', 'product']]
    :scaler:                sklearn scaler for normalization
    :drop_orig_cols:        drop original columns after count-encoding
    """

    # create temporary column with no missing values, used for counting
    df['tmp'] = 1
    
    new_cols = []
    
    if verbose: print('adding categorical counts...')
    for cols in column_combinations:
        # set name suffix for new column
        if isinstance(cols, list): var_name = '_'.join(cols)
        else:                      var_name = cols
        
        new_column_name = 'ft_' + var_name + '__count' 
        if verbose: print(new_column_name.ljust(60), end = ' ')
        
        # groupby count transform
        counts = df.groupby(cols)['tmp'].transform('count').values.reshape(-1, 1)#.astype(int)
        
        if scaler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     

                counts = scaler.fit_transform(counts); # suppress warnings
                scaler_str = str(type(scaler)).split('.')[-1].split('Scaler')[0].split('Transformer')[0].lower()
                new_column_name = f'{new_column_name}_scaled_{scaler_str}'
            
        df[new_column_name] = counts

        if verbose: print('unique', str( df[new_column_name].nunique() ).rjust(5), 
                          '| min',  str( df[new_column_name].min()     ).rjust(5),
                          '| max',  str( df[new_column_name].max()     ).rjust(5))
        
        if drop_orig_cols: df = df.drop(cols, axis=1)
        
        new_cols.append(new_column_name)
    
    df = df.drop('tmp', axis=1)
    
    if return_new_cols: return df, new_cols
    else:               return df


def create_label_encoding(df, column_combinations: list, min_count = 5, 
                       drop_orig_cols = False, return_new_cols = True, 
                       verbose = True):
    """
    Add numerical labels for categorical values.
    Values under a specified low total count are grouped together as '0'
    """
    #max_col_length = len(max(columns, key=len))
    
    new_cols = []
    
    df['tmp'] = 1

    if verbose: print('adding label encoding...')
    # set name suffix for new column
    for col_combo in column_combinations:
        col_str = '_'.join(col_combo) if isinstance(col_combo, list) else col_combo
        new_column_name = 'ft_' + col_str + '__label'
        new_cols.append(new_column_name)
        
        if verbose: print('-', new_column_name.ljust(50), end=' ')
        
        # if label encoding for column combination, use groupby.transform to get unique values per combination
        # if single column, use values directly
        if isinstance(col_combo, list):
            column_values = df.groupby(col_combo)[col_combo[0]].transform(lambda series: random.random())
        else:
            column_values = df[col_combo].copy()
                
        # determine low-count outliers, replace with '00000'
        col_counts = df.groupby(col_combo)['tmp'].transform("count")
        column_values = np.where(col_counts < min_count, '00000', column_values)

        # label encode remaining values
        label_encoder = preprocessing.LabelEncoder()
        df[new_column_name] = label_encoder.fit_transform(column_values)

        if verbose: print('unique:', str(df[new_column_name].nunique()).ljust(7))
        
        if drop_orig_cols: df = df.drop(col_combo, axis=1)
    
    df = df.drop('tmp', axis=1)

    if return_new_cols: return df, new_cols
    else:               return df


def create_one_hot_encoding(df, columns: list, min_pctg_to_keep=0.03, return_new_cols=True, verbose=True):
    """
    Adds one-hot encoded columns for each categorical column
    """
    max_col_length = len(max(columns, key=len))
    
    new_cols = []
    
    print('creating one-hot columns:')
    for column in columns:
        #df[column]    = df[column].apply(lambda x: str(x)) #convert to str just in case
        #new_columns = [column + "_" + i for i in full[column].unique()] #only use the columns that appear in the test set and add prefix like in get_dummies
        if verbose: print('-', column.ljust(max_col_length), end=' ')
        
        if df[column].nunique() > 500:
            print('too many unique values', df[column].nunique())
        else:    
            one_hot_df = pd.get_dummies(df[column], prefix=f'ft_{column}__one_hot_')
            orig_col_number = len(one_hot_df.columns)
            
            keep_cols  = (one_hot_df.sum()/len(one_hot_df))>=min_pctg_to_keep
            one_hot_df = one_hot_df.loc[:, keep_cols]

            if verbose: print(f'keep {len(one_hot_df.columns)}/{orig_col_number} one-hot columns')
    
            # drop columns if they already exist, in case function is called twice
            df = df.drop(one_hot_df.columns, axis=1, errors='ignore')
            df = pd.concat((df, one_hot_df), axis = 1)

            new_cols.extend(list(one_hot_df.columns))
    
    new_cols = list(set(new_cols))
    
    if return_new_cols: return df, new_cols
    else:               return df


def target_encode_smooth_mean(df, catg_columns:list, target_col:str, train_index, 
                              smoothing_factor=3, std_noise_factor=0.01, verbose=True):
    """
    Add smoothed mean target encoding.
    """
    max_col_length = len(max(catg_columns, key=len))    
    
    # Compute the global mean
    train_mean = df['target'].mean()
    print('global mean:', train_mean)
    
    for col in catg_columns:
        # Compute the number of values and the mean of each group for train data only
        grouped = df.loc[train_index, :].groupby(col)['target'].agg(['count', 'mean', 'std'])
        counts, means, stds = grouped['count'], grouped['mean'], grouped['std']
        
        # Compute the smoothed means
        smooth_mean = (counts*means + smoothing_factor*train_mean) / (counts + smoothing_factor)
        
        if isinstance(col, str):
            new_column_name = f'ft_{col}__target_enc_mean_smooth{smoothing_factor}'    
            df[new_column_name] = df[col].map(smooth_mean)

            # Add noise
            if std_noise_factor is not None:
                # add column with scaled standard deviation
                df['tmp_stds_with_noise'] = df[col].map(stds)*std_noise_factor

        elif isinstance(col, list):
            col_str = '_'.join(col)
            new_column_name = f'ft_{col_str}__target_enc_mean_smooth{smoothing_factor}'
            # remove column if already exist from previous execution of same function to prevent merge-duplicates
            df = df.drop(new_column_name, axis=1, errors='ignore') 
            
            smooth_mean_df = pd.DataFrame(smooth_mean).reset_index().rename(columns={0:new_column_name})
            df = pd.merge(df, smooth_mean_df, how='left', on=col)
            
            if std_noise_factor is not None:
                df = df.drop('tmp_stds_with_noise', axis=1, errors='ignore') 
                
                # add column with scaled standard deviation
                stds_df = pd.DataFrame(stds).reset_index().rename(columns={'std':'tmp_stds_with_noise'})
                df = pd.merge(df, stds_df, how='left', on=col)
                df['tmp_stds_with_noise'] = df['tmp_stds_with_noise']*std_noise_factor
        
        if std_noise_factor is not None:
            df['tmp_stds_with_noise'] = df['tmp_stds_with_noise'].fillna(0.1)
            # add random uniform noise
            np.random.seed(1)
            df.loc[train_index, 'tmp_stds_with_noise'] *= np.random.randn(len(train_index))
            df[new_column_name] = df[new_column_name] + df['tmp_stds_with_noise']

            df = df.drop(['tmp_stds_with_noise'], axis=1)
        

        if verbose and std_noise_factor is not None:
            print(f'added target encoding with noise ({std_noise_factor}*std):', new_column_name)
        elif verbose and std_noise_factor is None:
            print(f'added target encoding without noise:', new_column_name)

    return df