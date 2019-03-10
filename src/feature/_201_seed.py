"""season stat
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _201_Seed(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv')

    def create_feature_impl(self, df):
        feat = df.copy()
        feat['Seed'] = feat['Seed'].str[1:3].astype(int)

        return feat

    def post_process(self, trn, tst):
        trn['SeedDiff'] = trn['T1Seed'] - trn['T2Seed']
        tst['SeedDiff'] = tst['T1Seed'] - tst['T2Seed']
        return trn, tst


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
    print(df)
