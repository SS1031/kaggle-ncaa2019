"""Massey Ordinal
Kenneth Masseyさんが作っているランキング
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _401_POMRank(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'MasseyOrdinals.csv')

    def create_feature_impl(self, df):
        df = df[df.SystemName == 'POM'].sort_values(['Season', 'RankingDayNum', 'TeamID'])
        feat = df.groupby(['Season', 'TeamID']).agg({
            'OrdinalRank': ['min', 'max', 'mean', 'last', 'std']
        })

        feat.columns = ["_".join(x).replace('OrdinalRank', 'POMRank') for x in feat.columns.ravel()]

        return feat.reset_index()

    def post_process(self, trn, tst):
        trn['POM_last_diff'] = trn['T1POMRank_last'] - trn['T2POMRank_last']
        tst['POM_last_diff'] = tst['T1POMRank_last'] - tst['T2POMRank_last']

        return trn, tst


if __name__ == '__main__':
    trn, tst = _401_POMRank().create_feature(devmode=True)
