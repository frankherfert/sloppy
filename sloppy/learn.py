import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from fastprogress.fastprogress import progress_bar
import datetime
import gc


def set_learning_rate_with_resets(
    iteration, start=0.1, min_learning_rate=0.001, decay=0.99,
    reset_every=None, verbose=False):
    """
    LGB suitable learning rate decay
    Returns a decaying learning rate that will be reset to higher values at intervals.
    This can help to overcome local minima.

    Use with: learning_rates = lambda iter: set_learning_rate_with_resets()
    """
    if reset_every is not None:
        rate = max(min_learning_rate, round(start * (decay ** ((iteration % reset_every)+1)),6))
    else:
        rate = max(min_learning_rate, round(start * (decay ** iteration),6))

    if verbose>1 and iteration>0:
        if iteration%verbose==0: print('LR:', rate, end="\t")

    return rate


# def predict_out_of_fold_sklearn(est, n_splits, x_train, y, x_test):
#     oof_train = np.zeros((x_train.shape[0],))
#     oof_test  = np.zeros((x_test.shape[0] ,))
#     oof_test_skf = np.empty((n_splits, x_test.shape[0]))

#     print("gettings out of fold predictions for:\n",
#           "x_train", str(x_train.shape).ljust(20), type(x_train), "\n",
#           "y      ", str(y.shape      ).ljust(20), type(y),       "\n",
#           "x_test ", str(x_test.shape ).ljust(20), type(x_test),  "\n")

#     # if input is sparse, no need to transform
#     if type(x_train)==pd.core.frame.DataFrame:
#         x_train = x_train.values

#     if type(x_test)==pd.core.frame.DataFrame:
#         x_test = x_test.values

#     if type(y)==pd.core.frame.DataFrame or type(y)==pd.core.series.Series:
#         #print("y is df or series")
#         y = y.values.ravel()

#     folds = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=41
#                                  ).split(x_train, y)

#     for i, (train_index, test_index) in enumerate(folds):
#         print(str(datetime.datetime.now())[:19], "\t Fold:", str(i+1).rjust(3), end=", "
#              )
#         x_tr = x_train[train_index]
#         y_tr = y[train_index]
#         x_te = x_train[test_index]

#         est.fit(x_tr, y_tr)
#         print("training done", end=", ")

#         oof_train[test_index] = est.predict(x_te)
#         oof_test_skf[i, :] = est.predict(x_test)
#         print("predicting done")
#         gc.collect()

#     oof_test[:] = oof_test_skf.mean(axis=0)
#     gc.collect()
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def predict_out_of_fold_sklearn(df, train_index, predict_index, target:str, features:list,
                                n_splits:int = 5, preds_oof_col_suffix='_model_1',
                                est=None, model_init_params:dict = None, model_fit_params:dict = None,
                                verbose=True) -> pd.DataFrame:
    """
    Creates out of fold predictions using sklearn estimators
    Returns: source df with added predictions, models
    """

    models_trained = []

    # train_oof_preds = np.zeros(len(train_index))
    oof_preds_col = 'preds_oof_'+preds_oof_col_suffix
    df[oof_preds_col] = np.nan
    predict_preds_oof = pd.DataFrame(index=predict_index)

    folds = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=41
                                 ).split(df.loc[train_index, features],
                                         df.loc[train_index, target])

    for n_fold, (fold_train_index, fold_valid_index) in enumerate(progress_bar(list(folds)), start=1): # returns index independent of dataframe index
        # split into train and validation set
        x_train = df.loc[train_index, features].iloc[fold_train_index]
        x_valid = df.loc[train_index, features].iloc[fold_valid_index]
        y_train = df.loc[train_index, target  ].iloc[fold_train_index]
        y_valid = df.loc[train_index, target  ].iloc[fold_valid_index]

        print('train', x_train.shape, 'valid:', x_valid.shape, end='\t')
        if 'verbose' in model_fit_params: print()

        # init model
        model = est(**model_init_params)
        try:  # try to add eval set for early stopping
            if model.__repr__()[:4] == 'LGBM':
                model_fit_params['eval_set'] = (x_valid, y_valid)
                model_fit_params['eval_names'] = ('\t valid')
                model_fit_params['early_stopping_rounds'] = 50
            if model.__repr__().startswith('<catboost'):
                model_fit_params['eval_set'] = (x_valid, y_valid)
                model_fit_params['early_stopping_rounds'] = 20
        except: pass

        if model_fit_params is not None:
            model.fit(x_train, y_train, **model_fit_params)
        else: 
            model.fit(x_train, y_train)
        models_trained.append(model)

        # predictions
        train_preds_oof = model.predict(x_valid)
        predict_preds   = model.predict(df.loc[predict_index, features])
        
        df.loc[train_index[fold_valid_index], oof_preds_col] = train_preds_oof
        predict_preds_oof[f'pred_fold_{n_fold}'] = predict_preds
        
        if verbose:
            print(f'preds mean: {train_preds_oof.mean():.4f}', end=' | ')
            try: print('score:', model.best_score_)
            except: print()

    predict_preds_oof = predict_preds_oof.mean(axis=1)
    df.loc[predict_index, oof_preds_col] = predict_preds_oof

    return df, models_trained


def score_binary_predictions(y_true, y_pred):
    y_pred_01 = np.round(y_pred, 0)
    print("# \t",
          "roc auc ↑:",     round(metrics.roc_auc_score( y_true=y_true, y_score= y_pred),    4),
          "F1 ↑:",          round(metrics.f1_score(      y_true=y_true, y_pred=  y_pred_01), 4),
          "\t accuracy ↑:", round(metrics.accuracy_score(y_true=y_true, y_pred=  y_pred_01), 4),
          "\t log loss ↓:", round(metrics.log_loss(      y_true=y_true, y_pred=  y_pred),    4)
         )

