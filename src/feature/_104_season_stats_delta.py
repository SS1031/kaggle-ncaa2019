"""Stats delta from last year
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _104_SeasonStatsDelta(FeatureBase):
    fin = os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv")

    def create_feature_impl(self, df):
        tidy_df = tidy_detailed_data(df)
        tidy_df['DiffScore'] = tidy_df['Score'] - tidy_df['OppScore']
        uni_season = sorted(tidy_df.Season.unique())
        dict_agg = {
            "FGM": ['mean'],
            "FGA": ['mean'],
            "FGM3": ['mean'],
            "FGA3": ['mean'],
            "FTM": ['mean'],
            "FTA": ['mean'],
            "OR": ['mean'],
            "DR": ['mean'],
            "Ast": ['mean'],
            "TO": ['mean'],
            "Stl": ['mean'],
            "Blk": ['mean'],
            "PF": ['mean'],
            "DiffScore": ['mean'],
        }

        feat = pd.DataFrame()
        for s in uni_season:
            prev_s_agg = tidy_df[tidy_df['Season'] == (s - 1)].groupby('TeamID').agg(dict_agg)
            s_agg = tidy_df[tidy_df['Season'] == s].groupby('TeamID').agg(dict_agg)
            delta_s_agg = (s_agg - prev_s_agg)
            delta_s_agg.columns = ['Delta' + x[0] for x in delta_s_agg.columns.ravel()]
            delta_s_agg['Season'] = s
            feat = pd.concat([feat, delta_s_agg.reset_index()], axis=0).fillna(0).reset_index(drop=True)

        return feat

    def post_process(self, trn, tst):
        return trn, tst


def tidy_detailed_data(df):
    initial_nrow = df.shape[0]
    df['MatchID'] = (df['Season'].astype(str) + df['DayNum'].astype(str) +
                     df['WTeamID'].astype(str) + df['LTeamID'].astype(str))

    # データをTeamIDに対
    # るテーブルに変換する
    wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
    lcols = [c for c in df.columns if 'L' in c and c != 'WLoc']

    t1df = df[['MatchID', 'Season', 'DayNum'] + wcols + ['LScore', 'LOR', 'LDR', 'LFGA']].copy()
    rename_dict = dict(zip(wcols, [utils.lreplace('W', '', c) for c in wcols]))
    rename_dict['LScore'] = 'OppScore'
    rename_dict['LOR'] = 'OppOR'
    rename_dict['LDR'] = 'OppDR'
    rename_dict['LFGA'] = 'OppFGA'
    t1df.rename(columns=rename_dict, inplace=True)
    t1df['Result'] = 1

    t2df = df[['MatchID', 'Season', 'DayNum'] + lcols + ['WScore', 'WOR', 'WDR', 'WFGA']].copy()
    rename_dict = dict(zip(lcols, [utils.lreplace('L', '', c) for c in lcols]))
    rename_dict['WScore'] = 'OppScore'
    rename_dict['WOR'] = 'OppOR'
    rename_dict['WDR'] = 'OppDR'
    rename_dict['WFGA'] = 'OppFGA'
    t2df.rename(columns=rename_dict, inplace=True)
    t2df['Result'] = 0

    df = pd.concat([t1df, t2df], axis=0).reset_index(drop=True)

    assert (initial_nrow * 2) == df.shape[0]

    return df


if __name__ == '__main__':
    trn, tst = _104_SeasonStatsDelta().create_feature(devmode=True)
    print(trn.head())
    print(tst.head())
    # fin = os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv")
    # df = pd.read_csv(fin)
    # tidy_df = tidy_detailed_data(df)
    # tidy_df['DiffScore'] = tidy_df['Score'] - tidy_df['OppScore']
    # uni_season = sorted(tidy_df.Season.unique())
    #
    # dict_agg = {
    #     "FGM": ['mean'],
    #     "FGA": ['mean'],
    #     "FGM3": ['mean'],
    #     "FGA3": ['mean'],
    #     "FTM": ['mean'],
    #     "FTA": ['mean'],
    #     "OR": ['mean'],
    #     "DR": ['mean'],
    #     "Ast": ['mean'],
    #     "TO": ['mean'],
    #     "Stl": ['mean'],
    #     "Blk": ['mean'],
    #     "PF": ['mean'],
    # }
    #
    # feat = pd.DataFrame()
    # for s in uni_season[0:1]:
    #     prev_s_agg = tidy_df[tidy_df['Season'] == (s - 1)].groupby('TeamID').agg(dict_agg)
    #     s_agg = tidy_df[tidy_df['Season'] == s].groupby('TeamID').agg(dict_agg)
    #     delta_s_agg = (s_agg - prev_s_agg)
    #     delta_s_agg.columns = ['Delta' + x[0] for x in delta_s_agg.columns.ravel()]
    #     delta_s_agg['Season'] = s
    #     feat = pd.concat([feat, delta_s_agg.reset_index()], axis=0).fillna(0).reset_index(drop=True)
