import os
import pandas as pd

import CONST
from feature import FeatureBase


class _202_Conference(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'TeamConferences.csv')

    def create_feature_impl(self, df):
        df = df.rename(columns={'ConfAbbrev': 'Conference'})
        df['Conference'] = df['Conference'].astype('category')

        return df

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    trn, tst = _202_Conference().create_feature(devmode=True)
