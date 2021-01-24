import numpy as np
from sklearn import metrics
import datetime


def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def regression_metrics(y_true, y_pred, model_name:str):
    
    reg_metrics = {}
    
    reg_metrics['mse']  = metrics.mean_squared_error(y_true, y_pred)
    reg_metrics['rmse'] = metrics.mean_squared_error(y_true, y_pred, squared=False)
    reg_metrics['mae']  = metrics.median_absolute_error(y_true, y_pred)
    reg_metrics['r2']   = metrics.r2_score(y_true, y_pred)
    
    if y_true.min()<=0: pass
    else: reg_metrics['mape'] = mape(y_true, y_pred)
    
    reg_metrics['y_pred_<=0'] = (y_pred <= 0).sum()
    reg_metrics['y_pred_max'] = y_pred.max()
    reg_metrics['std'] = y_pred.std()
    
    return reg_metrics


def regression_metrics(y_true, y_pred, model_name:str):
    mse  = metrics.mean_squared_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    mae  = metrics.median_absolute_error(y_true, y_pred)
    try: y_mape = mape(y_true, y_pred)
    except: y_mape = None; #print('div by zero')
    r2   = metrics.r2_score(y_true, y_pred)

    y_true_zero_or_neg = (y_true <= 0).sum()
    y_pred_zero_or_neg = (y_pred <= 0).sum()
    
    text =  f'# {model_name} - MSE↓: {mse:,.4f} | RMSE↓: {rmse:,.4f} | MAE↓: {mae:,.4f} | '
    if y_mape is not None: text += f'MAPE↓: {y_mape:.4%} | R2↑: {r2:.4f}'
    
    text += f'\n# - y_true: avg: {y_true.mean():.4} | std: {y_true.std():,.2f} | '
    text += f'<=0: {y_true_zero_or_neg:,} | min: {y_true.min():,.2f} | '
    text += f'max: {y_true.max():,.2f} '

    text += f'\n# - y_pred: avg: {y_pred.mean():.4} | std: {y_pred.std():,.2f} | '
    text += f'<=0: {y_pred_zero_or_neg:,} | min: {y_pred.min():,.2f} | '
    text += f'max: {y_pred.max():,.2f} '

    return text

###############################################################################

def binary_metrics(y_true, y_pred) -> dict:
    y_pred_01 = np.round(y_pred,0)

    bin_metrics = {}
    bin_metrics['roc auc ↑'] =  round(metrics.roc_auc_score( y_true=y_true, y_score= y_pred),    4)
    bin_metrics['F1 ↑'] =       round(metrics.f1_score(      y_true=y_true, y_pred=  y_pred_01), 4)
    bin_metrics['accuracy ↑'] = round(metrics.accuracy_score(y_true=y_true, y_pred=  y_pred_01), 4)
    bin_metrics['log loss ↓'] = round(metrics.log_loss(      y_true=y_true, y_pred=  y_pred),    4)
    
    return bin_metrics


def binary_metrics_text(y_true, y_pred, timestamp=True) -> str:
    bin_metrics_dict = binary_metrics(y_true, y_pred)
    
    text = '#'
    for key, value in bin_metrics_dict.items():
        text += f'{key}: {value}\t'
        
    if timestamp: text+= str(datetime.datetime.now())[:19]
    
    return text