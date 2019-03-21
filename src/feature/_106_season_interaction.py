"""シーズン対戦成績
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _106_SeasonInteraction(FeatureBase):
    fin = os.path.join(CONST.INDIR, "RegularSeasonCompactResults.csv")

    def create_feature_impl(self, df):
        # Dummy
        tidy_df = utils.tidy_data(df)
        return tidy_df.groupby(['Season', 'TeamID']).size().reset_index()[['Season', 'TeamID']]

    def post_process(self, trn, tst):
        trn = trn.drop(columns=['Result'])

        trn_initial_row = len(trn)
        tst_initial_row = len(tst)

        df = pd.read_csv(os.path.join(CONST.INDIR, "RegularSeasonCompactResults.csv"))[['Season', 'WTeamID', 'LTeamID']]
        t1_rename_dict = {'WTeamID': 'TeamID', 'LTeamID': 'OppTeamID'}
        t2_rename_dict = {'LTeamID': 'TeamID', 'WTeamID': 'OppTeamID'}

        season_interaction = pd.DataFrame()
        season_history_interaction = pd.DataFrame()

        for s in [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]:
            t1df = df[df.Season <= s].copy()
            t1df.rename(columns=t1_rename_dict, inplace=True)
            tmp1 = t1df.groupby(['TeamID', 'OppTeamID']).size().to_frame('SeasonHistoryInterWinCnt')
            t2df = df[df.Season <= s].copy()
            t2df.rename(columns=t2_rename_dict, inplace=True)
            tmp2 = t2df.groupby(['TeamID', 'OppTeamID']).size().to_frame('SeasonHistoryInterLoseCnt')

            tmp = pd.concat([tmp1, tmp2], axis=1).fillna(0)
            tmp['SeasonHistoryInteraction'] = tmp['SeasonHistoryInterWinCnt'] - tmp['SeasonHistoryInterLoseCnt']
            tmp['Season'] = s
            season_history_interaction = pd.concat([season_history_interaction, tmp], axis=0)

            t1df = df[df.Season == s].copy()
            t1df.rename(columns=t1_rename_dict, inplace=True)
            tmp1 = t1df.groupby(['TeamID', 'OppTeamID']).size().to_frame('SeasonInterWinCnt')
            t2df = df[df.Season == s].copy()
            t2df.rename(columns=t2_rename_dict, inplace=True)
            tmp2 = t2df.groupby(['TeamID', 'OppTeamID']).size().to_frame('SeasonInterLoseCnt')

            tmp = pd.concat([tmp1, tmp2], axis=1).fillna(0)
            tmp['SeasonInteraction'] = tmp['SeasonInterWinCnt'] - tmp['SeasonInterLoseCnt']
            tmp['Season'] = s

            season_interaction = pd.concat([season_interaction, tmp], axis=0)

        season_history_interaction = season_history_interaction.reset_index().rename(columns={'TeamID': 'T1TeamID',
                                                                                              'OppTeamID': 'T2TeamID'})
        season_interaction = season_interaction.reset_index().rename(columns={'TeamID': 'T1TeamID',
                                                                              'OppTeamID': 'T2TeamID'})

        trn = trn.merge(season_history_interaction, on=['Season', 'T1TeamID', 'T2TeamID'], how='left').fillna(0)
        assert trn_initial_row == len(trn)
        trn = trn.merge(season_interaction, on=['Season', 'T1TeamID', 'T2TeamID'], how='left').fillna(0)
        assert trn_initial_row == len(trn)
        tst = tst.merge(season_history_interaction, on=['Season', 'T1TeamID', 'T2TeamID'], how='left').fillna(0)
        assert tst_initial_row == len(tst)
        tst = tst.merge(season_interaction, on=['Season', 'T1TeamID', 'T2TeamID'], how='left').fillna(0)
        assert tst_initial_row == len(tst)
        return trn, tst


if __name__ == '__main__':
    trn, tst = _106_SeasonInteraction().create_feature(devmode=True)
