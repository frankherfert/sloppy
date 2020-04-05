import pandas as pd
import numpy as np

from sklearn import model_selection


def create_folds_column(df, train_index=None, n_folds:int=5, stratified=False, verbose=True) -> pd.DataFrame:
    """
    Creates a new column with fold numbers.
    
    Args:
        train_index: Row index of training data. Uses entire index if None.
        n_folds (int): Number of folds. Must be at least 2.
        stratified (bool, optional): Use StratifiedKFold, expects column 'target' in df.
        verbose (bool): Print basic statistics for each fold.
    """
    
    if stratified: fold = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:          fold = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)

    df['fold'] = -1
    
    train_index = df.index if train_index is not None

    folds_gen = fold.split(X=df.loc[train_index, :],
                           y=df.loc[train_index, 'target'].values)
    
    for fold_nr, (train_index, val_index) in enumerate(folds_gen, start=1):
        df.loc[val_index, 'fold'] = fold_nr
    
    if verbose:
        df.groupby(['fold']).agg({'target':['count', 'mean', 'median', 'std']})
    
    return df