"""Touney win history
"""
import os
import pandas as pd

import CONST
from feature import FeatureBase


class _301_TourneyWins(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'NCAATourneyCompactResults.csv')

    def create_feature_impl(self, df):
        df['Season'] = df['Season'] - 1
        uni_season = df.Season.unique()
        feat = pd.DataFrame()
        for s in uni_season:
            tmp = df[df['Season'] < s + 1].groupby('WTeamID').agg({'WTeamID': 'size'}).rename(
                columns={'WTeamID': 'TourneyWins'}).reset_index()
            tmp['Season'] = s + 1
            feat = pd.concat([feat, tmp], axis=0).reset_index(drop=True)

        feat.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
        return feat

    def post_process(self, trn, tst):
        trn['T1TourneyWins'].fillna(0, inplace=True)
        trn['T2TourneyWins'].fillna(0, inplace=True)
        tst['T1TourneyWins'].fillna(0, inplace=True)
        tst['T2TourneyWins'].fillna(0, inplace=True)

        return trn, tst


if __name__ == '__main__':
    trn, tst = _301_TourneyWins().create_feature(devmode=True)
