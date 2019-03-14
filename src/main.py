import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument('--tuning', action='store_true')
    options = parser.parse_args()

    with open(options.config, "r") as fp:
        conf = json.load(fp, object_pairs_hook=OrderedDict)

    config_name = os.path.basename(options.config).replace(".json", "")

    SEED = conf['seed']
    np.random.seed(SEED)

    trn, tst = load_feature_sets(options.config)

    sbmt = pd.read_csv(os.path.join(CONST.INDIR, CONST.STAGE1))
    sbmt.drop(columns=['Pred'], inplace=True)
    tmp = sbmt.ID.str.split('_', expand=True).astype(int)
    tmp.columns = ['Season', 'T1TeamID', 'T2TeamID']
    sbmt = pd.concat([sbmt, tmp], axis=1)


    def validate_and_pred(trn, tst):
        feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
        valid_season = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        valid_scores = []
        df_preds = pd.DataFrame(np.empty((tst.shape[0], 10)))

        for s in valid_season:
            print(f"Split Season : {s}")
            for i in range(10):
                seed = SEED + i ** 10
                np.random.seed(seed)
                train = trn[trn.Season < s].sample(frac=0.9, random_state=seed)
                valid = trn[s == trn.Season]
                d_train = lgb.Dataset(train[feature_cols].astype(np.float32),
                                      label=train['Result'].values, feature_name=feature_cols)
                d_valid = lgb.Dataset(valid[feature_cols].astype(np.float32),
                                      label=valid['Result'].values, feature_name=feature_cols)
                model = lgb.train({'objective': 'binary', 'seed': seed}, d_train,
                                  num_boost_round=10000,
                                  valid_sets=[d_valid], early_stopping_rounds=100,
                                  verbose_eval=100)
                valid_scores.append(model.best_score['valid_0']['binary_logloss'])
                if (s + 1) in tst.Season.unique():
                    df_preds.loc[tst.Season == (s + 1), i] = model.predict(
                        tst[tst.Season == (s + 1)][feature_cols])

        print(f'Validation Score {np.mean(valid_scores)} +-({np.std(valid_scores)})')
        return np.mean(valid_scores), df_preds


    score, df_preds = validate_and_pred(trn, tst)
    sbmt = sbmt.merge(pd.concat([tst[['Season', 'T1TeamID', 'T2TeamID']],
                                 df_preds.mean(axis=1).to_frame('Pred')], axis=1),
                      on=['Season', 'T1TeamID', 'T2TeamID'], how='left')
    sbmt.loc[sbmt.Pred <= 0.025, 'Pred'] = 0.025
    sbmt.loc[sbmt.Pred >= 0.975, 'Pred'] = 0.975

    ### Anomaly event happened only once before - be brave
    # seed = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
    # seed['Seed'] = seed['Seed'].str[1:3].astype(int)
    # sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T1TeamID', 'Seed': 'T1Seed'}),
    #                   on=['Season', 'T1TeamID'], how='left')
    # sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T2TeamID', 'Seed': 'T2Seed'}),
    #                   on=['Season', 'T2TeamID'], how='left')
    # sbmt.loc[(sbmt.T1Seed == 16) & (sbmt.T2Seed == 1), 'Pred'] = 0.025
    # sbmt.loc[(sbmt.T1Seed == 15) & (sbmt.T2Seed == 2), 'Pred'] = 0.025
    # sbmt.loc[(sbmt.T1Seed == 14) & (sbmt.T2Seed == 3), 'Pred'] = 0.025
    # sbmt.loc[(sbmt.T1Seed == 13) & (sbmt.T2Seed == 4), 'Pred'] = 0.025
    # sbmt.loc[(sbmt.T1Seed == 1) & (sbmt.T2Seed == 16), 'Pred'] = 0.975
    # sbmt.loc[(sbmt.T1Seed == 2) & (sbmt.T2Seed == 15), 'Pred'] = 0.975
    # sbmt.loc[(sbmt.T1Seed == 3) & (sbmt.T2Seed == 14), 'Pred'] = 0.975
    # sbmt.loc[(sbmt.T1Seed == 4) & (sbmt.T2Seed == 13), 'Pred'] = 0.975


    sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, config_name + '.csv'), index=False)
    ans = utils.load_trn_base()
    ans = sbmt.merge(ans[['Season', 'T1TeamID', 'T2TeamID', 'Result']],
                     on=['Season', 'T1TeamID', 'T2TeamID'], how='inner')
    print('logloss(after post-process)', log_loss(ans['Result'], ans['Pred']))
