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

import CONST
import feature._001_utils as utils
from feature._002_load import load_feature_sets
from sklearn.metrics import log_loss

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
    # from selection import cor_selector

    # save_path = cor_selector(options.config)
    trn, tst = load_feature_sets(conf_file=options.config)


    def validate_and_pred(trn, tst, iteration=10, params={'objective': 'binary'}, predict=True, verbose=True):
        feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
        valid_season = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
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
                d_train = lgb.Dataset(train[feature_cols].astype(np.float32),
                                      label=train['Result'].values, feature_name=feature_cols)
                d_valid = lgb.Dataset(valid[feature_cols].astype(np.float32),
                                      label=valid['Result'].values, feature_name=feature_cols)
                model = lgb.train(params, d_train,
                                  num_boost_round=10000,
                                  valid_sets=[d_valid],
                                  early_stopping_rounds=100,
                                  verbose_eval=verbose * 100)
                valid_scores.append(model.best_score['valid_0']['binary_logloss'])
                if (s + 1) in tst.Season.unique():
                    df_preds.loc[tst.Season == (s + 1), i] = model.predict(
                        tst[tst.Season == (s + 1)][feature_cols])

        sbmt = pd.read_csv(os.path.join(CONST.INDIR, CONST.STAGE1))
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

    score, sbmt = validate_and_pred(trn, tst, iteration=10, params=params, predict=True, verbose=True)
    sbmt.loc[sbmt.Pred <= 0.025, 'Pred'] = 0.025
    sbmt.loc[sbmt.Pred >= 0.975, 'Pred'] = 0.975
    sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, config_name + '.csv'), index=False)
    sbmt[sbmt.Season == 2018][['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, '2018_' + config_name + '.csv'),
                                                     index=False)

    # if options.postprocess:
    #     print("Do post processing, Be Brave")
    #     ### Anomaly event happened only once before - be brave
    #     seed = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
    #     seed['Seed'] = seed['Seed'].str[1:3].astype(int)
    #     sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T1TeamID', 'Seed': 'T1Seed'}),
    #                       on=['Season', 'T1TeamID'], how='left')
    #     sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T2TeamID', 'Seed': 'T2Seed'}),
    #                       on=['Season', 'T2TeamID'], how='left')
    #     sbmt.loc[(sbmt.T1Seed == 16) & (sbmt.T2Seed == 1), 'Pred'] = 0
    #     sbmt.loc[(sbmt.T1Seed == 15) & (sbmt.T2Seed == 2), 'Pred'] = 0
    #     sbmt.loc[(sbmt.T1Seed == 14) & (sbmt.T2Seed == 3), 'Pred'] = 0.025
    #     sbmt.loc[(sbmt.T1Seed == 13) & (sbmt.T2Seed == 4), 'Pred'] = 0.025
    #     sbmt.loc[(sbmt.T1Seed == 1) & (sbmt.T2Seed == 16), 'Pred'] = 1
    #     sbmt.loc[(sbmt.T1Seed == 2) & (sbmt.T2Seed == 15), 'Pred'] = 1
    #     sbmt.loc[(sbmt.T1Seed == 3) & (sbmt.T2Seed == 14), 'Pred'] = 0.975
    #     sbmt.loc[(sbmt.T1Seed == 4) & (sbmt.T2Seed == 13), 'Pred'] = 0.975
    #     sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, 'pp' + config_name + '.csv'), index=False)
    #     sbmt[sbmt.Season == 2018][['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, 'pp2018_' + config_name + '.csv'),
    #                                                      index=False)
    #     ans = utils.load_trn_base()
    #     ans = sbmt.merge(ans[['Season', 'T1TeamID', 'T2TeamID', 'Result']],
    #                      on=['Season', 'T1TeamID', 'T2TeamID'], how='inner')
    #     print('logloss(after post-process)', log_loss(ans['Result'], ans['Pred']))

    # 63%|█████████████████████████████▌                 | 63/100 [00:06<00:03,  9.69 sec/trial]
    # [2019-03-16 21:28:13,829] Finished a trial resulted in value: 0.9999. Current best value is 1.0000
