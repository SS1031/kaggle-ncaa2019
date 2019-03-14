"""Massey Ordinal
Kenneth Masseyさんが作っているランキング
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _402_SagarinRank(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'MasseyOrdinals.csv')

    def create_feature_impl(self, df):
        df = df[df.SystemName == 'SAG'].sort_values(['Season', 'RankingDayNum', 'TeamID'])
        feat = df.groupby(['Season', 'TeamID']).agg({
            'OrdinalRank': ['min', 'max', 'mean', 'last', 'std']
        })

        feat.columns = ["_".join(x).replace('OrdinalRank', 'SagarinRank') for x in feat.columns.ravel()]

        return feat.reset_index()

    def post_process(self, trn, tst):
        trn['Sagarin_last_diff'] = trn['T1SagarinRank_last'] - trn['T2SagarinRank_last']
        tst['Sagarin_last_diff'] = tst['T1SagarinRank_last'] - tst['T2SagarinRank_last']

        return trn, tst


if __name__ == '__main__':
    trn, tst = _402_SagarinRank().create_feature(devmode=True)
    pd.read_csv(os.path.join(CONST.INDIR, 'MasseyOrdinals.csv'))
