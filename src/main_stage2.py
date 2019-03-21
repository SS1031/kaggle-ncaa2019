import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from collections import OrderedDict
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
    parser.add_argument('--selected', default='')
    options = parser.parse_args()
    with open(options.config, "r") as fp:
        conf = json.load(fp, object_pairs_hook=OrderedDict)

    config_name = os.path.basename(options.config).replace(".json", "")
    SEED = conf['seed']
    np.random.seed(SEED)

    options = parser.parse_args()

    # from selection import cor_selector
    # # save_path = cor_selector(options.config)

    if options.selected != '':
        trn, tst = load_feature_sets(selected_feature_file=options.selected)
    else:
        trn, tst = load_feature_sets(conf_file=options.config)


    def seed_average(trn, tst, iteration=10, params={'objective': 'binary'}, predict=True, verbose=True):

        feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
        categorical_cols = trn.select_dtypes('category').columns.tolist()

        valid_season = [2013, 2014, 2015, 2016, 2017, 2018]
        valid_scores = []
        df_preds = pd.DataFrame(np.empty((tst[tst.Season.isin([2014, 2015, 2016, 2017, 2018, 2019])].shape[0],
                                          iteration)))
        df_preds2019 = tst[tst.Season == 2019][['Season', 'T1TeamID', 'T2TeamID']]
        feature_importance_df = pd.DataFrame()

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

                _score = model.best_score['valid_0']['binary_logloss']
                _preds = model.predict(tst[tst.Season == (s + 1)][feature_cols])
                _preds2019 = model.predict(tst[tst.Season == 2019][feature_cols])

                valid_scores.append(_score)
                df_preds.loc[tst.Season == (s + 1), i] = _preds
                df_preds2019[f"{s}_{i}"] = _preds2019

                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = feature_cols
                fold_importance_df["importance"] = model.feature_importance()
                fold_importance_df["fold"] = s + i
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        sbmt = pd.read_csv(CONST.SS)
        sbmt.drop(columns=['Pred'], inplace=True)
        tmp = sbmt.ID.str.split('_', expand=True).astype(int)
        tmp.columns = ['Season', 'T1TeamID', 'T2TeamID']
        sbmt = pd.concat([sbmt, tmp], axis=1)
        sbmt = sbmt.merge(pd.concat([tst[['Season', 'T1TeamID', 'T2TeamID']],
                                     df_preds.mean(axis=1).to_frame('Pred')], axis=1),
                          on=['Season', 'T1TeamID', 'T2TeamID'], how='left')

        sbmt2 = pd.read_csv(os.path.join(CONST.INDIR, 'SampleSubmissionStage2.csv')).drop(columns=['Pred'])
        sbmt2 = pd.concat([sbmt2, sbmt2.ID.str.split('_', expand=True).astype(int)], axis=1)
        sbmt2.columns = ['ID', 'Season', 'T1TeamID', 'T2TeamID']
        sbmt2 = sbmt2.merge(pd.concat([
            df_preds2019[['Season', 'T1TeamID', 'T2TeamID']],
            df_preds2019[
                [c for c in df_preds2019 if c not in ['Season', 'T1TeamID', 'T2TeamID']]
            ].mean(axis=1).to_frame('Pred')
        ], axis=1), on=['Season', 'T1TeamID', 'T2TeamID'], how='left')

        ans = utils.load_trn_base()
        ans = sbmt.merge(ans[['Season', 'T1TeamID', 'T2TeamID', 'Result']],
                         on=['Season', 'T1TeamID', 'T2TeamID'], how='inner')

        if predict:
            print(f'Validation Score {np.mean(valid_scores)} +-({np.std(valid_scores)})')
            print('logloss', log_loss(ans['Result'], ans['Pred']))
            return (log_loss(ans['Result'], ans['Pred']), sbmt[sbmt.Season == 2019].reset_index(drop=True),
                    sbmt2, feature_importance_df)
        else:
            return log_loss(ans['Result'], ans['Pred'])


    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'bagging_fraction': 0.9,
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

            return seed_average(trn, tst, iteration=1, params=params, predict=False, verbose=False)


    objective = Objective(trn, tst)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    params['num_leaves'] = study.best_params['num_leaves']
    params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    params['max_bin'] = study.best_params['max_bin']
    params['bagging_fraction'] = study.best_params['bagging_fraction']
    params['learning_rate'] = 0.008

    score, sbmt, sbmt2, feature_importance_df = seed_average(trn, tst, iteration=10, params=params,
                                                             predict=True, verbose=True)
    assert pd.read_csv(os.path.join(CONST.INDIR, 'SampleSubmissionStage2.csv'))['ID'].equals(sbmt['ID'])
    assert pd.read_csv(os.path.join(CONST.INDIR, 'SampleSubmissionStage2.csv'))['ID'].equals(sbmt2['ID'])

    sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, config_name + '.csv'), index=False)
    sbmt2[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, 'SeasonAvg_' + config_name + '.csv'), index=False)

    cols = (feature_importance_df[
                ["feature", "importance"]
            ].groupby("feature").mean().sort_values(by="importance",
                                                    ascending=False)[:100].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(CONST.OUTDIR, f'imp_{config_name}.png'))

    print("Do post processing, Be Brave")
    seed = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
    seed['Seed'] = seed['Seed'].str[1:3].astype(int)
    sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T1TeamID', 'Seed': 'T1Seed'}),
                      on=['Season', 'T1TeamID'], how='left')
    sbmt = sbmt.merge(seed.rename(columns={'TeamID': 'T2TeamID', 'Seed': 'T2Seed'}),
                      on=['Season', 'T2TeamID'], how='left')
    sbmt.loc[(sbmt.T1Seed == 16) & (sbmt.T2Seed == 1), 'Pred'] = 0
    sbmt.loc[(sbmt.T1Seed == 15) & (sbmt.T2Seed == 2), 'Pred'] = 0
    sbmt.loc[(sbmt.T1Seed == 14) & (sbmt.T2Seed == 3), 'Pred'] = 0
    sbmt.loc[(sbmt.T1Seed == 1) & (sbmt.T2Seed == 16), 'Pred'] = 1
    sbmt.loc[(sbmt.T1Seed == 2) & (sbmt.T2Seed == 15), 'Pred'] = 1
    sbmt.loc[(sbmt.T1Seed == 3) & (sbmt.T2Seed == 14), 'Pred'] = 1
    sbmt[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, 'PP_' + config_name + '.csv'), index=False)

    sbmt2 = sbmt2.merge(seed.rename(columns={'TeamID': 'T1TeamID', 'Seed': 'T1Seed'}),
                        on=['Season', 'T1TeamID'], how='left')
    sbmt2 = sbmt2.merge(seed.rename(columns={'TeamID': 'T2TeamID', 'Seed': 'T2Seed'}),
                        on=['Season', 'T2TeamID'], how='left')
    sbmt2.loc[(sbmt2.T1Seed == 16) & (sbmt2.T2Seed == 1), 'Pred'] = 0
    sbmt2.loc[(sbmt2.T1Seed == 15) & (sbmt2.T2Seed == 2), 'Pred'] = 0
    sbmt2.loc[(sbmt2.T1Seed == 14) & (sbmt2.T2Seed == 3), 'Pred'] = 0
    sbmt2.loc[(sbmt2.T1Seed == 1) & (sbmt2.T2Seed == 16), 'Pred'] = 1
    sbmt2.loc[(sbmt2.T1Seed == 2) & (sbmt2.T2Seed == 15), 'Pred'] = 1
    sbmt2.loc[(sbmt2.T1Seed == 3) & (sbmt2.T2Seed == 14), 'Pred'] = 1
    # GoZags
    sbmt2.loc[(sbmt2.T1TeamID == 1211), 'Pred'] = 1
    sbmt2.loc[(sbmt2.T2TeamID == 1211), 'Pred'] = 0
    sbmt2[['ID', 'Pred']].to_csv(os.path.join(CONST.SBMTDIR, 'PP_SeasonAVG_' + config_name + '.csv'), index=False)
