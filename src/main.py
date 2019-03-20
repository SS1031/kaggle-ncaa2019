import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

import CONST
import feature._001_utils as utils
from feature._002_load import load_feature_sets

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/config001.debug.json')
    parser.add_argument('--postprocess', action='store_true')
    options = parser.parse_args()
    with open(options.config, "r") as fp:
        conf = json.load(fp, object_pairs_hook=OrderedDict)

    config_name = os.path.basename(options.config).replace(".json", "")
    SEED = conf['seed']
    np.random.seed(SEED)

    options = parser.parse_args()

    trn, tst = load_feature_sets(conf_file=options.config)

    def validate_and_pred(trn, tst, iteration=10, params={'objective': 'binary'}, predict=True, verbose=True):
        feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
        categorical_cols = trn.select_dtypes('category').columns.tolist()
        valid_season = [2013, 2014, 2015, 2016, 2017]
        valid_scores = []
        df_preds = pd.DataFrame(np.empty((tst.shape[0], iteration)))
        for s in valid_season:
            if verbose: print(f"Split Season : {s}")
            for i in range(iteration):
                seed = (SEED + i) ** 2
                np.random.seed(seed)
                params['seed'] = seed
                params['bagging_seed'] = seed
                train = trn[trn.Season < s]
                valid = trn[s == trn.Season]

                d_train = lgb.Dataset(train[feature_cols],
                                      label=train['Result'].values,
                                      feature_name=feature_cols,
                                      categorical_feature=categorical_cols)

                d_valid = lgb.Dataset(valid[feature_cols],
                                      label=valid['Result'].values,
                                      feature_name=feature_cols,
                                      categorical_feature=categorical_cols)

                model = lgb.train(params, d_train,
                                  num_boost_round=10000,
                                  valid_sets=[d_valid],
                                  early_stopping_rounds=100,
                                  verbose_eval=verbose * 100)
                valid_scores.append(model.best_score['valid_0']['binary_logloss'])
                if (s + 1) in tst.Season.unique():
                    df_preds.loc[tst.Season == (s + 1), i] = model.predict(
                        tst[tst.Season == (s + 1)][feature_cols])


        sbmt = pd.read_csv(CONST.SS)
        sbmt.drop(columns=['Pred'], inplace=True)
        tmp = sbmt.ID.str.split('_', expand=True).astype(int)
        tmp.columns = ['Season', 'T1TeamID', 'T2TeamID']
        sbmt = pd.concat([sbmt, tmp], axis=1)
        sbmt = sbmt.merge(pd.concat([tst[['Season', 'T1TeamID', 'T2TeamID']],
                                     df_preds.mean(axis=1).to_frame('Pred')], axis=1),
                          on=['Season', 'T1TeamID', 'T2TeamID'], how='left')

        ans = utils.load_trn_base()
        ans = sbmt.merge(ans[['Season', 'T1TeamID', 'T2TeamID', 'Result']],
                         on=['Season', 'T1TeamID', 'T2TeamID'], how='inner')
        print(ans)
        if verbose:
            print(f'Validation Score {np.mean(valid_scores)} +-({np.std(valid_scores)})')
            print('logloss', log_loss(ans['Result'], ans['Pred']))
        if predict:
            return log_loss(ans['Result'], ans['Pred']), sbmt
        else:
            return log_loss(ans['Result'], ans['Pred'])


    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'bagging_freq': 1,
    }


    class Objective(object):
        def __init__(self, trn, tst):
            self.trn = trn
            self.tst = tst

        def __call__(self, trial):
            trn, tst = self.trn, self.tst

            params['num_leaves'] = trial.suggest_int('num_leaves', 10, 100)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 50)
            params['max_bin'] = trial.suggest_int('max_bin', 64, 512)
            params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            # params['lambda_l1'] = trial.suggest_uniform('lambda_l1', 0.7, 1.0)
            # params['lambda_l2'] = trial.suggest_uniform('lambda_l2', 0.7, 1.0)
            params['verbose'] = -1

            return validate_and_pred(trn, tst, iteration=1, params=params, predict=False, verbose=False)


    objective = Objective(trn, tst)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    params['num_leaves'] = study.best_params['num_leaves']
    params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    params['max_bin'] = study.best_params['max_bin']
    params['bagging_fraction'] = study.best_params['bagging_fraction']
    params['learning_rate'] = 0.01

    score, sbmt = validate_and_pred(trn, tst, iteration=1, params=params, predict=True, verbose=True)
    sbmt.loc[sbmt.Pred <= 0.025, 'Pred'] = 0.025
    sbmt.loc[sbmt.Pred >= 0.975, 'Pred'] = 0.975
    # sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, config_name + '.csv'), index=False)
    # sbmt[sbmt.Season == 2018][['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, '2018_' + config_name + '.csv'),
    #                                                  index=False)

