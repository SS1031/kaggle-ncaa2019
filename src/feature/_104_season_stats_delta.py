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
        tidy_df = utils.tidy_detailed_data(df)
        tidy_df['ScoreDiff'] = tidy_df['Score'] - tidy_df['OppScore']
        tidy_df['ORDiff'] = tidy_df['OR'] - tidy_df['OppOR']
        tidy_df['DRDiff'] = tidy_df['DR'] - tidy_df['OppDR']
        tidy_df['TODiff'] = tidy_df['TO'] - tidy_df['OppTO']
        tidy_df['FGM2'] = tidy_df['FGM'] - tidy_df['FGM3']
        tidy_df['FGA2'] = tidy_df['FGA'] - tidy_df['FGA3']

        agg_dict = {
            'Score': ['mean'],
            'FGM': ['mean'],
            'FGM2': ['mean'],
            'FGM3': ['mean'],
            'FGA': ['mean'],
            'FGA2': ['mean'],
            'FGA3': ['mean'],
            'FTM': ['mean'],
            'FTA': ['mean'],
            'OR': ['mean'],
            'DR': ['mean'],
            'Ast': ['mean'],
            'TO': ['mean'],
            'Stl': ['mean'],
            'Blk': ['mean'],
            'PF': ['mean'],
            'OppScore': ['mean'],
            'OppOR': ['mean'],
            'OppDR': ['mean'],
            'OppFGA': ['mean'],
            'OppTO': ['mean'],
            'Result': ['mean'],
        }

        uni_season = sorted(tidy_df.Season.unique().tolist())

        feat = pd.DataFrame()
        for s in uni_season:
            prev_s_agg = tidy_df[tidy_df['Season'] == (s - 1)].groupby('TeamID').agg(agg_dict)
            s_agg = tidy_df[tidy_df['Season'] == s].groupby('TeamID').agg(agg_dict)
            delta_s_agg = (s_agg - prev_s_agg)
            delta_s_agg.columns = ['Delta' + x[0] for x in delta_s_agg.columns.ravel()]
            delta_s_agg['Season'] = s
            feat = pd.concat([feat, delta_s_agg.reset_index()], axis=0).fillna(0).reset_index(drop=True)

        return feat

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    trn, tst = _104_SeasonStatsDelta().create_feature(devmode=True)
