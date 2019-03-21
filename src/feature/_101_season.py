"""season stat
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _101_RegularSeasonStats(FeatureBase):
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
            'Score': ['mean', 'median'],
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
            'OppScore': ['mean', 'median'],
            'OppOR': ['mean'],
            'OppDR': ['mean'],
            'OppFGA': ['mean'],
            'OppTO': ['mean'],
            'Result': ['mean', 'sum'],
        }

        feat = tidy_df.groupby(['Season', 'TeamID']).agg(agg_dict)
        feat.columns = ["_".join(x) for x in feat.columns.ravel()]

        feat14d = tidy_df[tidy_df.DayNum > 118].groupby(['Season', 'TeamID']).agg(agg_dict)
        feat14d.columns = [x[0] + '14D_' + x[1] for x in feat14d.columns.ravel()]

        feat = pd.concat([feat, feat14d], axis=1).reset_index()

        return feat

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    train, test = _101_RegularSeasonStats().create_feature(devmode=True)
