import os
import json
import argparse
import numpy as np
import pandas as pd

from collections import OrderedDict

import CONST
from feature._002_load import load_feature_sets


def cor_selector(config):
    config_name = os.path.basename(config).replace(".json", "")

    trn, tst = load_feature_sets(conf_file=config)

    feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
    X = trn[feature_cols].copy()
    y = trn['Result'].copy()

    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_cols]

    print("selected feautre num ", len(cor_feature))
    save_path = os.path.join(CONST.SELECTEDDIR, f'cor_selection_{config_name}.csv')
    pd.DataFrame({'feature': cor_feature}).to_csv(save_path, index=False)

    return save_path


def rf_selector(config):
    config_name = os.path.basename(config).replace(".json", "")

    trn, tst = load_feature_sets(conf_file=config)

    feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
    X = trn[feature_cols].copy()
    y = trn['Result'].copy()
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()

    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()

    print(str(len(embeded_rf_feature)), 'selected features')


def lgbm_select(config):
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMClassifier

    config_name = os.path.basename(config).replace(".json", "")
    trn, tst = load_feature_sets(conf_file=config)
    feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
    X = trn[feature_cols].copy()
    y = trn['Result'].copy()

    lgbc = LGBMClassifier()
    embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
    embeded_lgb_selector.fit(X, y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features')

    save_path = os.path.join(CONST.SELECTEDDIR, f'lgb_selection_{config_name}.csv')
    pd.DataFrame({'feature': embeded_lgb_feature}).to_csv(save_path, index=False)

    return embeded_lgb_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/config001.debug.json')

    options = parser.parse_args()
    # save_path = cor_selector(options.config)
    # trn, tst = load_feature_sets(conf_file=options.config)
    # trn, tst = load_feature_sets(conf_file='./config/config010.json')
    # rf_selector(options.config)
    feature = lgbm_select(options.config)
