import pandas as pd
import numpy as np
from sklearn import model_selection
import datetime
import gc


def set_learning_rate_with_resets(iteration, start=0.1, min_learning_rate=0.001, decay=0.99, reset_every=None, verbose=False):
    """
    LGB suitable learning rate decay
    Returns a decaying learning rate that will be reset to higher values at intervals.
    This can help to overcome local minima.

    Use with: learning_rates = lambda iter: set_learning_rate_with_resets()
    """
    if reset_every!=None:
        rate = max(min_learning_rate, round(start * (decay ** ((iteration%reset_every)+1)),6))
    else:
        rate = max(min_learning_rate, round(start * (decay ** iteration),6))

    if verbose>1 and iteration>0:
        if iteration%verbose==0: print('LR:', rate, end="\t")

    return rate


def get_out_of_fold_train_test_predictions(clf, kfolds, x_train, y, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros( (x_test.shape[0] ,))
    oof_test_skf = np.empty((kfolds.n_folds, x_test.shape[0]))
    print("gettings out of fold predictions for:\n",
          "x_train", str(x_train.shape).ljust(20), type(x_train), "\n",
          "y      ", str(y.shape).ljust(20),       type(y),       "\n",
          "x_test ", str(x_test.shape).ljust(20),  type(x_test),  "\n")

    # if input is sparse, no need to transform
    
    if type(x_train)==pd.core.frame.DataFrame:
        x_train = x_train.values
    
    if type(x_test)==pd.core.frame.DataFrame:
        x_test = x_test.values
    
    if type(y)==pd.core.frame.DataFrame or type(y)==pd.core.series.Series:
        #print("y is df or series")
        y = y.values.ravel()
        
    for i, (train_index, test_index) in enumerate(kfolds):
        print(str(datetime.datetime.now())[:19], "\t Fold:", i+1, #end=", "
             )
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        print("training done", end=", ")
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        print("predicting done")
        gc.collect()
    
    oof_test[:] = oof_test_skf.mean(axis=0)
    gc.collect()
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def predict_out_of_fold(df, train_index, predict_index, target:str, features:list, n_splits:int = 5, oof_preds_col_suffix='_model_1',
                        est=None, model_init_parameters:dict = None, model_fit_parameters:dict = None):
    """
    Creates out of fold predictions using LightGBM.
    Returns: source df with added predictions, model_importances
    """
    
    model = est(**model_init_parameters)
    model_importances = pd.DataFrame({'feature':features})

    #train_oof_preds = np.zeros(len(train_index))
    oof_preds_col = 'oof_preds'+oof_preds_col_suffix
    df[oof_preds_col] = np.nan
    pred_oof_preds = pd.DataFrame(index=predict_index)

    folds = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=41
                                 ).split(df.loc[train_index, features],
                                         df.loc[train_index, target])

    for n_fold, (fold_train_index, fold_valid_index) in enumerate(folds, start=1): # returns index independent of dataframe index        
        # split into train and validation set
        X_train = df.loc[train_index, features].iloc[fold_train_index]
        X_valid = df.loc[train_index, features].iloc[fold_valid_index]
        y_train = df.loc[train_index, target  ].iloc[fold_train_index]
        y_valid = df.loc[train_index, target  ].iloc[fold_valid_index]
        
        print(str(n_fold).rjust(2)+'/'+str(n_splits), 
              'train', X_train.shape, 'valid:', X_valid.shape, #'valid_index:', fold_valid_index,
              end='\t')
        
        # train on train-part of dataset
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], #eval_names=('\t test') #[(X_train, y_train), (X_valid, y_valid)], 
                  **model_fit_parameters)
        print('early stopping:', model.best_iteration_)

        model_importances[f'imp_{n_fold}'] = model.feature_importances_

        # assign predictions to validation part
        oof_preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        df.loc[train_index[fold_valid_index], oof_preds_col] = oof_preds
        #df.loc[train_index[fold_valid_index], 'train_oof_fold' ] = n_fold
        
        # assign predictions to predict part
        pred_oof_preds[f'pred_fold_{n_fold}'] = model.predict(df.loc[predict_index, features], num_iteration=model.best_iteration_)

    model_importances['imp_mean'] = model_importances.mean(axis=1)
    model_importances['pctg']     = np.round(100 * model_importances['imp_mean'] / model_importances['imp_mean'].sum(), 4)
    model_importances = model_importances.sort_values('imp_mean', ascending=False).reset_index(drop=True)

    pred_oof_preds = pred_oof_preds.mean(axis=1)
    df.loc[predict_index, oof_preds_col] = pred_oof_preds
    
    return df, model_importances






