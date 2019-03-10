"""season stat
"""
import os
import sys
import numpy as np
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase

from scipy import stats


def wrap_linear_trend(y):
    return stats.linregress(np.arange(len(y)), y.values)


class _102_RegularSeasonEloRating(FeatureBase):
    fin = os.path.join(CONST.INDIR, "RegularSeasonCompactResults.csv")

    def create_feature_impl(self, df):
        """https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings
        """
        K = 20.
        HOME_ADVANTAGE = 100.

        team_ids = set(df.WTeamID).union(set(df.LTeamID))

        # This dictionary will be used as a lookup for current
        # scores while the algorithm is iterating through each game
        elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

        # Elo updates will be scaled based on the margin of victory
        df['margin'] = df.WScore - df.LScore

        def elo_pred(elo1, elo2):
            return (1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

        def expected_margin(elo_diff):
            return ((7.5 + 0.006 * elo_diff))

        def elo_update(w_elo, l_elo, margin):
            elo_diff = w_elo - l_elo
            pred = elo_pred(w_elo, l_elo)
            mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
            update = K * mult * (1 - pred)
            return (pred, update)

        # I'm going to iterate over the games dataframe using
        # index numbers, so want to check that nothing is out
        # of order before I do that.
        assert np.all(df.index.values == np.array(range(df.shape[0]))), "Index is out of order."
        preds = []
        w_elo = []
        l_elo = []

        # Loop over all rows of the games dataframe
        for row in df.itertuples():
            # Get key data from current row
            w = row.WTeamID
            l = row.LTeamID
            margin = row.margin
            wloc = row.WLoc

            # Does either team get a home-court advantage?
            w_ad, l_ad, = 0., 0.
            if wloc == "H":
                w_ad += HOME_ADVANTAGE
            elif wloc == "A":
                l_ad += HOME_ADVANTAGE

            # Get elo updates as a result of the game
            pred, update = elo_update(elo_dict[w] + w_ad, elo_dict[l] + l_ad, margin)
            elo_dict[w] += update
            elo_dict[l] -= update

            # Save prediction and new Elos for each round
            preds.append(pred)
            w_elo.append(elo_dict[w])
            l_elo.append(elo_dict[l])

        df['WELO'] = w_elo
        df['LELO'] = l_elo
        wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
        lcols = [c for c in df.columns if 'L' in c]
        df1 = df[['Season', 'DayNum', 'WTeamID', 'WELO', 'LELO']].copy().rename(
            columns={'WTeamID': 'TeamID', 'WELO': 'ELO', 'LELO': 'OppELO'}
        )
        df2 = df[['Season', 'DayNum', 'LTeamID', 'LELO', 'WELO']].copy().rename(
            columns={'LTeamID': 'TeamID', 'LELO': 'ELO', 'WELO': 'OppELO'}
        )

        df = pd.concat([df1, df2], axis=0).sort_values(['Season', 'DayNum', 'TeamID']).reset_index(drop=True)
        df['DiffELO'] = df.ELO - df.OppELO
        stats = df.groupby(['Season', 'TeamID']).agg({'ELO': ['min', 'mean', 'max', 'std'],
                                                      'DiffELO': ['min', 'mean', 'max', 'std']})
        stats.columns = ["_".join(x) for x in stats.columns.ravel()]
        ltrend = df.groupby(['Season', 'TeamID'])['ELO'].apply(wrap_linear_trend)
        ltrend = pd.DataFrame(ltrend.values.tolist(),
                              columns=['ELO_slope',
                                       'ELO_intercept',
                                       'ELO_r_value',
                                       'ELO_p_value',
                                       'ELO_std_err'],
                              index=ltrend.index)
        df14D = df[df.DayNum > 118]  # 最後の14日間
        stats14D = df14D.groupby(['Season', 'TeamID']).agg({'ELO': ['min', 'mean', 'max'],
                                                            'DiffELO': ['min', 'mean', 'max', 'std']})
        stats14D.columns = ["_".join(x).replace("ELO", 'ELO14D') for x in stats14D.columns.ravel()]
        ltrend14D = df14D.groupby(['Season', 'TeamID'])['ELO'].apply(wrap_linear_trend)
        ltrend14D = pd.DataFrame(ltrend14D.values.tolist(),
                                 columns=['ELO14D_slope',
                                          'ELO14D_intercept',
                                          'ELO14D_r_value',
                                          'ELO14D_p_value',
                                          'ELO14D_std_err'],
                                 index=ltrend14D.index).fillna(0)
        feat = pd.concat([stats, stats14D, ltrend, ltrend14D], axis=1).fillna(0).reset_index()

        return feat

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    train, test = _102_RegularSeasonEloRating().create_feature(devmode=True)
