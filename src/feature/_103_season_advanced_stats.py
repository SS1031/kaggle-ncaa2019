"""season stat
References:
- https://www.kaggle.com/humburgc/history-eda-machine-learning-march-madness
"""
import os
import sys
import pandas as pd

import CONST
import feature._001_utils as utils
from feature import FeatureBase


class _103_SeasonAdvancedStats(FeatureBase):
    fin = os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv")

    def create_feature_impl(self, df):
        tidy_df = utils.tidy_detailed_data(df)

        # Possessions = 0.96 x (FGA + Turnovers + (0.475 x FTA) - Offensive Rebounds
        tidy_df['Poss'] = 0.96 * (tidy_df['FGA'] + tidy_df['TO'] + ((0.475) * tidy_df['FTA']) - tidy_df['OR'])
        # Offensive Rating = 100 x (Score / Possessions)
        tidy_df['OffRtg'] = 100 * (tidy_df['Score'] / tidy_df['Poss'])
        # Defensive Rating = 100 x (Opponent's Score / Possessions)
        tidy_df['DefRtg'] = 100 * (tidy_df['OppScore'] / tidy_df['Poss'])
        # Net Rating = 100 x (Offensive Rating - Defensive Rating)
        tidy_df['NetRtg'] = 100 * (tidy_df['OffRtg'] - tidy_df['DefRtg'])
        # Assist ratio = 100 * Assists / (FGA + (0.475 * FTA) + Asistst + Turnovers))
        tidy_df['AstRto'] = (100 * tidy_df['Ast'] /
                             (tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO']))
        # Turnover ratio = 100 * Turnovers / (FGA + (0.475 * FTA) + Assists + Turnovers)
        tidy_df['TORto'] = (100 * tidy_df['TO'] /
                            (tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO']))
        # True Shooting % = 100 * Team Points / (2 * (FGA + (0.475 * FTA)))
        tidy_df['ShtPct'] = 100 * tidy_df['Score'] / (2 * (tidy_df['FGA'] + (0.475 * tidy_df['FTA'])))
        # Effective FG% = (FGM + 0.5 * Threes Made) / FGA
        tidy_df['EffFGPct'] = (tidy_df['FGM'] + 0.5 * tidy_df['FGM3']) / tidy_df['FGA']
        # Free Throw rate = FTA / FGA
        tidy_df['FTR'] = tidy_df['FTA'] / tidy_df['FGA']
        # Offensive Rebound % = Offensive Rebounds / (Offensive Rebounds + Opponent's Defensive Rebounds)
        tidy_df['ORPct'] = tidy_df['OR'] / (tidy_df['OR'] + tidy_df['OppOR'])
        # Defensive Rebound % = Defensive Rebounds / (Defensive Rebounds + Opponent's Offensive Rebounds)
        tidy_df['DRPct'] = tidy_df['DR'] / (tidy_df['DR'] + tidy_df['OppDR'])
        # Total Rebound % = (Defensive Rebounds + Offensive Rebounds) /
        #                   (Defensive Rebounds + Offensive Rebounds
        #                   + Opponent's Defensive Rebounds + Opponent's Offensive Rebounds)
        tidy_df['TRPct'] = ((tidy_df['DR'] + tidy_df['OR']) /
                            (tidy_df['DR'] + tidy_df['OR'] + tidy_df['OppDR'] + tidy_df['OppOR']))
        # Turn Over per Possession = (Turnovers / possessions)
        tidy_df['TOpPos'] = (tidy_df['TO'] / tidy_df['Poss'])
        # Free Throw Rate
        tidy_df['FTRtg'] = (tidy_df['FTM'] / tidy_df['FGA'])
        # Team Impact Estimate =
        #   (Score + FGM + FTM - FGA - FTA + DR + 0.5*OR + Ast + Stl + 0.5*Blk - PF - TO)
        tidy_df['ImpEst'] = (tidy_df['Score'] + tidy_df['FGM'] + tidy_df['FTM'] - tidy_df['FGA'] -
                             tidy_df['FTA'] + tidy_df['DR'] + .5 * tidy_df['OR'] + tidy_df['Ast'] +
                             tidy_df['Stl'] + .5 * tidy_df['Blk'] - tidy_df['PF'])
        tidy_df['OppImpEst'] = (tidy_df['OppScore'] + tidy_df['OppFGM'] + tidy_df['OppFTM'] - tidy_df['OppFGA'] -
                                tidy_df['OppFTA'] + tidy_df['OppDR'] + .5 * tidy_df['OppOR'] + tidy_df['OppAst'] +
                                tidy_df['OppStl'] + .5 * tidy_df['OppBlk'] - tidy_df['OppPF'])
        tidy_df['IE'] = tidy_df['ImpEst'] / (tidy_df['ImpEst'] + tidy_df['OppImpEst'])
        # Block Percentage = Blk / OppFGA2
        tidy_df['OppFGA2'] = tidy_df['OppFGA'] - tidy_df['OppFGA3']
        tidy_df['BlkPct'] = tidy_df['Blk'] / tidy_df['OppFGA2']
        # Steel Percentage = Stl / OppPoss
        tidy_df['OppPoss'] = 0.96 * (tidy_df['OppFGA'] + tidy_df['OppTO'] +
                                     ((0.475) * tidy_df['OppFTA']) - tidy_df['OppOR'])
        tidy_df['StlPct'] = tidy_df['Stl'] / tidy_df['OppPoss']

        agg_dict = {
            'Poss': ['mean', 'median'],
            'OffRtg': ['mean'],
            'DefRtg': ['mean'],
            'NetRtg': ['mean'],
            'AstRto': ['mean'],
            'TORto': ['mean'],
            'ShtPct': ['mean'],
            'EffFGPct': ['mean'],
            'FTR': ['mean'],
            'ORPct': ['mean'],
            'DRPct': ['mean'],
            'TRPct': ['mean'],
            'TOpPos': ['mean'],
            'FTRtg': ['mean'],
            'IE': ['mean'],
            'BlkPct': ['mean'],
            'StlPct': ['mean'],
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
    train, test = _103_SeasonAdvancedStats().create_feature(devmode=True)

    # df = pd.read_csv(os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv"))
    # tidy_df = utils.tidy_detailed_data(df)
