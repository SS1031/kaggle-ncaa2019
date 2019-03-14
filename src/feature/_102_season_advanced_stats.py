"""season stat
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _102_SeasonAdvancedStats(FeatureBase):
    fin = os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv")

    def create_feature_impl(self, df):
        # データをTeamIDに対するテーブルに変換する
        wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
        lcols = [c for c in df.columns if 'L' in c]

        rename_dict = dict(zip(wcols, [utils.lreplace('W', 'T1', c) for c in wcols]))
        rename_dict.update(dict(zip(lcols, [utils.lreplace('L', 'T2', c) for c in lcols])))
        rename_dict.update({'WLoc': 'T1Home'})
        df1 = df.copy().rename(columns=rename_dict)
        df1['Result'] = 1
        df1['T1Home'] = (df1['T1Home'] == 'H')
        rename_dict = dict(zip(lcols, [utils.lreplace('L', 'T1', c) for c in lcols]))
        rename_dict.update(dict(zip(wcols, [utils.lreplace('W', 'T2', c) for c in wcols])))
        rename_dict.update({'WLoc': 'T1Home'})
        df2 = df.copy().rename(columns=rename_dict)
        df2['T1Home'] = (df2['T1Home'] == 'A')
        df2['Result'] = 0
        season = pd.concat([df1, df2], axis=0)

        season['DiffScore'] = season['T1Score'] - season['T2Score']
        feat = pd.concat([
            season.groupby(['Season', 'T1TeamID']).agg({
                'T1Score': ['mean', 'median'],
                'T2Score': ['mean'],
                'DiffScore': ['mean'],
                'T1FGA': ['mean', 'median', 'min', 'max'],
                'T1Ast': ['mean'],
                'T1Blk': ['mean'],
                'T2FGA': ['mean', 'min'],
            }),
        ])
        feat.columns = ["_".join(x) for x in feat.columns.ravel()]
        feat = pd.concat([
            feat,
            (season[season.DayNum > 118].groupby(['Season', 'T1TeamID']).Result.sum() /
             season[season.DayNum > 118].groupby(['Season', 'T1TeamID']).size()).rename('T1WinRatio14D'),
            season[season.DayNum > 118].groupby(['Season', 'T1TeamID']).T1Score.mean().rename('T1Score14D_mean')
        ], axis=1).fillna(0).reset_index()
        rename_dict = dict(zip(feat.columns, [c.replace('T1', '') for c in feat.columns]))
        feat.rename(columns=rename_dict, inplace=True)
        rename_dict = dict(zip(feat.columns, [c.replace('T2', 'Opp') for c in feat.columns]))
        feat.rename(columns=rename_dict, inplace=True)

        return feat

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    # train, test = _101_RegularSeasonStats().create_feature(devmode=False)

    df = pd.read_csv(os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv"))

    df['MatchID'] = (df['Season'].astype(str) + df['DayNum'].astype(str) +
                     df['WTeamID'].astype(str) + df['LTeamID'].astype(str))

    # データをTeamIDに対するテーブルに変換する
    wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
    lcols = [c for c in df.columns if 'L' in c and c != 'WLoc']

    t1df = df[['MatchID', 'Season', 'DayNum'] + wcols].copy()
    rename_dict = dict(zip(wcols, [utils.lreplace('W', '', c) for c in wcols]))
    t1df.rename(columns=rename_dict, inplace=True)
    t1df['Result'] = 1
    t1df['IsHome'] = (df['WLoc'] == 'H')

    t2df = df[['MatchID', 'Season', 'DayNum'] + lcols].copy()
    rename_dict = dict(zip(lcols, [utils.lreplace('L', '', c) for c in lcols]))
    t2df.rename(columns=rename_dict, inplace=True)
    t2df['Result'] = 0
    t2df['IsHome'] = (df['WLoc'] == 'A')

    df = pd.concat([t1df, t2df], axis=0)

    # Possessions = 0.96 x (FGA + Turnovers + (0.475 x FTA) - Offensive Rebounds
    df['Poss'] = 0.96 * (df['FGA'] + df['TO'] + ((0.475) * df['FTA']) - df['OR'])
