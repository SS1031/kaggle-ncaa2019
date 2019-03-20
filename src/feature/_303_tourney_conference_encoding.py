import os
import pandas as pd

import CONST
from feature import FeatureBase
import feature._001_utils as utils


class _303_TourneyConferenceEncoding(FeatureBase):
    fin = [
        os.path.join(CONST.INDIR, 'TeamConferences.csv'),
        os.path.join(CONST.INDIR, 'NCAATourneyCompactResults.csv')
    ]

    def create_feature_impl(self, df_list):
        df_conference = df_list[0]
        df_tourney = df_list[1]

        df_t1 = df_tourney.rename(columns={'WTeamID': 'TeamID'})[['Season', 'TeamID']]
        df_t1['Result'] = 1
        df_t2 = df_tourney.rename(columns={'LTeamID': 'TeamID'})[['Season', 'TeamID']]
        df_t2['Result'] = 0
        df = pd.concat([df_t1, df_t2], axis=0)

        df = df.merge(df_conference.rename(columns={'ConfAbbrev': 'Conference'}),
                      on=['Season', 'TeamID'], how='left')
        df['Season'] = df['Season'] + 1
        uni_season = df.Season.unique()
        feat = pd.DataFrame()

        for s in uni_season:
            tmp = df[df['Season'] <= s].groupby('Conference').agg({'Result': 'mean'}).rename(
                columns={'Result': 'ConferenceWinMean'}).reset_index()
            tmp['Season'] = s
            feat = pd.concat([feat, tmp], axis=0).reset_index(drop=True)

        # df['Season'] = df['Season'] + 1
        feat = df_conference.rename(columns={'ConfAbbrev': 'Conference'}).merge(
            feat, on=['Season', 'Conference'], how='left')[['Season', 'TeamID', 'ConferenceWinMean']]

        return feat

    def post_process(self, trn, tst):
        trn.fillna(0, inplace=True)
        tst.fillna(0, inplace=True)
        return trn, tst


if __name__ == '__main__':
    trn, tst = _303_TourneyConferenceEncoding().create_feature(devmode=True)
    df_conference = pd.read_csv(os.path.join(CONST.INDIR, 'TeamConferences.csv'))
