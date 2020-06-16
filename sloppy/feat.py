import pandas as pd
import numpy as np
import eli5
import shap


def feat_importances_from_models(models:list, features:list):
    model_importances = pd.DataFrame({'feature':features})

    for counter, m in enumerate(models):
        model_importances[f'imp_{counter}'] = m.feature_importances_

    model_importances['imp_mean'] = model_importances.mean(axis=1)
    model_importances['pctg']     = np.round(100 * model_importances['imp_mean'] / model_importances['imp_mean'].sum(), 4)

    model_importances = model_importances.sort_values('imp_mean', ascending=False).reset_index(drop=True)

    return model_importances


def eli5_importance(est, importance_type='gain', rank=False) -> pd.DataFrame:
    """
    Creates a df with all features and their importance.
    importance_type: gain, split
    """
    eli5_imp = eli5.format_as_dataframe(
        eli5.lightgbm.explain_weights_lightgbm(est, 
                                               importance_type=importance_type, 
                                               top=est.n_features_
                                              ))
    eli5_imp = eli5_imp.rename(columns={'weight': 'eli5_imp'})
    eli5_imp['eli5_imp'] = (eli5_imp['eli5_imp'] / eli5_imp['eli5_imp'].sum())*100

    if rank:
        eli5_imp['eli5_imp_rank'] = eli5_imp['eli5_imp'].rank(method='average', 
                                                              ascending=False
                                                              ).astype(int)
    
    return eli5_imp


def permutation_importance(est, feat_df, target, n_iter=1, rank=False, verbose=True) -> pd.DataFrame:
    """
    Creates a df with all features and their importance from a permutation-shuffle run.
    n_iter: Number of random shuffles for each feature, higher: more accurate, slower
    rank:   add column with rank of importance
    """
    if verbose: print('calculating permutation importance...', end=' ')

    perm_imp = (eli5.sklearn.PermutationImportance(est, random_state=42, n_iter=n_iter)
                .fit(feat_df, target)
               )

    perm_imp_df = (pd.DataFrame({'feature': feat_df.columns.to_list(), 
                                 'perm_imp': perm_imp.feature_importances_}
                               )
                   .sort_values('perm_imp', ascending=False)
                   .reset_index(drop=True)
                   )
    perm_imp_df['perm_imp'] = (perm_imp_df['perm_imp'] / perm_imp_df['perm_imp'].sum())*100
    
    if rank:
        perm_imp_df['perm_imp_rank'] = perm_imp_df['perm_imp'].rank(method='average', 
                                                                    ascending=False
                                                                   ).astype(int)
    if verbose: print('done, features with positive importance:', len(perm_imp_df.query('perm_imp > 0')))
    
    return perm_imp_df


def shap_values_df(est=None, feat_df=None, features:list=None, target=None, verbose=True) -> pd.DataFrame:
    """
    Creates a df with shap values for each feature.
    Can take multiple minutes larger iput dataframes.
    """
    explainer = shap.TreeExplainer(model=est,
                                   data=feat_df[features],
                                   feature_dependence='independent',
                                   model_output='probability')
    if verbose: 
        try:    print('expected value (mean/base prediction):', round(explainer.expected_value[0], 2))
        except: print('expected value (mean/base prediction):', round(explainer.expected_value,    2))

    # create array of shap values for each row
    shap_values = explainer.shap_values(feat_df[features], target)
    
    # create df with shap values and real column names
    shap_df = pd.DataFrame(data=shap_values,
                           index=feat_df.index,
                           columns=features
                          )
    shap_df.columns.name = None
    
    return shap_df


def combine_feats_and_shap_values(feat_df, shap_df, index_name='index', verbose=True):
    """
    Combine features and shapley values in one long dataframe.
    """
    # turn feat_df into long format
    feat_df_long = (feat_df[shap_df.columns]
                    .reset_index()
                    .melt(id_vars=index_name, var_name='feature', value_name='feature_value')
                   )
    if verbose: print('merging:', feat_df_long.shape, end=' + ')
    
    # turn shap_df into long format
    shap_df_long = (shap_df
                    .reset_index(drop=False)
                    .melt(id_vars=index_name, var_name='feature', value_name='shap_value')
                   )
    shap_df_long['shap_value_abs'] = shap_df_long['shap_value'].abs()
    shap_df_long['shap_value_rank'] = (shap_df_long.groupby([index_name])['shap_value_abs']
                                       .rank(method='average', ascending=False)
                                       .astype(int)
                                      )
    shap_df_long = shap_df_long.drop('shap_value_abs', axis=1)
    if verbose: print(shap_df_long.shape, end=' -> ')
    #    shap_df_long['shap_value']     = np.round(shap_df_long['shap_value'], 2)

    # merge two dataframes, sort by index
    feat_shap_df = pd.merge(feat_df_long, shap_df_long, how='inner', on=[index_name, 'feature'])
    if verbose: print(feat_shap_df.shape)
    feat_shap_df = feat_shap_df.sort_values([index_name, 'shap_value_rank'], 
                                            ascending=[True, True]).reset_index(drop=True)
    
    return feat_shap_df


def shap_importance(shap_df, rank=False) -> pd.DataFrame:
    """
    Creates a df with all features and their importance.
    """
    data = shap_df.abs().mean().sort_values(ascending=False)

    shap_imp = (pd.DataFrame(data=data, columns=['shap_imp'])
                .reset_index()
                .rename(columns={'index':'feature'})
               )
    
    shap_imp['shap_imp'] = (shap_imp['shap_imp'] / shap_imp['shap_imp'].sum())*100

    if rank:
        shap_imp['shap_imp_rank'] = shap_imp['shap_imp'].rank(method='average', 
                                                              ascending=False
                                                             ).astype(int)
    
    return shap_imp