import os
import re
import numpy as np
import pandas as pd

import CONST
import feature._001_utils as utils

d_season = pd.read_csv(os.path.join(CONST.INDIR, 'RegularSeasonDetailedResults.csv'))
d_tourney = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyDetailedResults.csv'))
sub = pd.read_csv(os.path.join(CONST.INDIR, 'SampleSubmissionStage1.csv'))
seed = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))

# データを勝ち負けにするため2倍にする
d_seas_wcols = [c for c in d_season.columns if 'W' in c and c != 'WLoc']
d_seas_lcols = [c for c in d_season.columns if 'L' in c]
rename_dict = dict(zip(d_seas_wcols, [utils.lreplace('W', 'T1', c) for c in d_seas_wcols]))
rename_dict.update(dict(zip(d_seas_lcols, [utils.lreplace('L', 'T2', c) for c in d_seas_lcols])))
rename_dict.update({'WLoc': 'T1Home'})
d_seas1 = d_season.copy().rename(columns=rename_dict)
d_seas1['Result'] = 1
d_seas1['T1Home'] = (d_seas1['T1Home'] == 'H')
rename_dict = dict(zip(d_seas_lcols, [utils.lreplace('L', 'T1', c) for c in d_seas_lcols]))
rename_dict.update(dict(zip(d_seas_wcols, [utils.lreplace('W', 'T2', c) for c in d_seas_wcols])))
rename_dict.update({'WLoc': 'T1Home'})
d_seas2 = d_season.copy().rename(columns=rename_dict)
d_seas2['T1Home'] = (d_seas2['T1Home'] == 'A')
d_seas2['Result'] = 0
d_seas_double = pd.concat([d_seas1, d_seas2], axis=0)

# データを勝ち負けにするため2倍にする
# d_tour_wcols = [c for c in d_tourney.columns if 'W' in c and c != 'WLoc']
# d_tour_lcols = [c for c in d_tourney.columns if 'L' in c]
# rename_dict = dict(zip(d_tour_wcols, [lreplace('W', 'T1', c) for c in d_tour_wcols]))
# rename_dict.update(dict(zip(d_tour_lcols, [lreplace('L', 'T2', c) for c in d_tour_lcols])))
# rename_dict.update({'WLoc': 'T1Home'})
# d_tour1 = d_tourney.copy().rename(columns=rename_dict)
# d_tour1['Result'] = 1
# d_tour1['T1Home'] = (d_tour1['T1Home'] == 'H')
# rename_dict = dict(zip(d_tour_lcols, [lreplace('L', 'T1', c) for c in d_tour_lcols]))
# rename_dict.update(dict(zip(d_tour_wcols, [lreplace('W', 'T2', c) for c in d_tour_wcols])))
# rename_dict.update({'WLoc': 'T1Home'})
# d_tour2 = d_tourney.copy().rename(columns=rename_dict)
# d_tour2['T1Home'] = (d_tour2['T1Home'] == 'A')
# d_tour2['Result'] = 0
trn_base = utils.load_trn_base()

# d_seas_double = d_seas_double[d_seas_double[['T1Seed', 'T2Seed']].notnull().all(axis=1)]

# Regular Season Stats
d_seas_double['DiffScore'] = d_seas_double['T1Score'] - d_seas_double['T2Score']
seas_stats = pd.concat([
    d_seas_double.groupby(['Season', 'T1TeamID']).agg({
        'T1Score': ['mean', 'median'],
        'T2Score': ['mean'],
        'DiffScore': ['mean'],
        'T1FGA': ['mean', 'median', 'min', 'max'],
        'T1Ast': ['mean'],
        'T1Blk': ['mean'],
        'T2FGA': ['mean', 'min'],
    }),
])
seas_stats.columns = ["_".join(x) for x in seas_stats.columns.ravel()]
seas_stats = pd.concat([
    seas_stats,
    (d_seas_double[d_seas_double.DayNum > 118].groupby(['Season', 'T1TeamID']).Result.sum() /
     d_seas_double[d_seas_double.DayNum > 118].groupby(['Season', 'T1TeamID']).size()).rename('T1WinRatio14D'),
    d_seas_double[d_seas_double.DayNum > 118].groupby(['Season', 'T1TeamID']).T1Score.mean().rename('T1Score14D_mean')
], axis=1).fillna(0).reset_index()

rename_dict = dict(zip(seas_stats.columns, [c.replace('T1', '') for c in seas_stats.columns]))
seas_stats.rename(columns=rename_dict, inplace=True)
rename_dict = dict(zip(seas_stats.columns, [c.replace('T2', 'Opp') for c in seas_stats.columns]))
seas_stats.rename(columns=rename_dict, inplace=True)

seas_stats_T1 = pd.concat([
    seas_stats[['Season', 'TeamID']].rename(columns={'TeamID': 'T1TeamID'}),
    seas_stats[[c for c in seas_stats.columns if c not in ['Season', 'TeamID']]].add_prefix('T1')
], axis=1)
seas_stats_T2 = pd.concat([
    seas_stats[['Season', 'TeamID']].rename(columns={'TeamID': 'T2TeamID'}),
    seas_stats[[c for c in seas_stats.columns if c not in ['Season', 'TeamID']]].add_prefix('T2')
], axis=1)

trn_base = trn_base.merge(seas_stats_T1, on=['Season', 'T1TeamID'], how='left')
trn_base = trn_base.merge(seas_stats_T2, on=['Season', 'T2TeamID'], how='left')
seed['Seed'] = seed['Seed'].str[1:3].astype(int)
trn_base = trn_base.merge(seed.rename(columns={'Seed': 'T1Seed', 'TeamID': 'T1TeamID'}),
                          on=['Season', 'T1TeamID'], how='left')
trn_base = trn_base.merge(seed.rename(columns={'Seed': 'T2Seed', 'TeamID': 'T2TeamID'}),
                          on=['Season', 'T2TeamID'], how='left')
trn_base['SeedDiff'] = trn_base.T1Seed - trn_base.T2Seed

import lightgbm as lgb

feature_name = [c for c in trn_base.columns if c not in ['Season', 'T1TeamID', 'T2TeamID', 'Result']]
X = trn_base[feature_name].values.astype(np.float32)
y = trn_base.Result.astype(int)

# # foldsの番号振る
# N = len(d_tour_double)
# n_fold = 10
# folds = []
# for i in range(1, n_fold):
#     folds += [i] * math.floor(N / n_fold)
# folds += [n_fold] * (N - (n_fold - 1) * math.floor(N / n_fold))


tst_base = utils.load_tst_base()
tst_base = tst_base.merge(seas_stats_T1, on=['Season', 'T1TeamID'], how='left')
tst_base = tst_base.merge(seas_stats_T2, on=['Season', 'T2TeamID'], how='left')

from sklearn.model_selection import StratifiedKFold

for i in range(10):
    seed = i ** 10
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
    for trn_ix, vld_ix in skf.split(X, y):
        d_trn = lgb.Dataset(X[trn_ix], label=y[trn_ix], feature_name=feature_name)
        d_vld = lgb.Dataset(X[vld_ix], label=y[vld_ix], feature_name=feature_name)
        eval_hist = lgb.train({'objective': 'binary'}, d_trn, num_boost_round=10000,
                              valid_sets=[d_vld], early_stopping_rounds=100,
                              verbose_eval=100)
