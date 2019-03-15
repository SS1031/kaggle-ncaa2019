"""season stat
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
        df['MatchID'] = (df['Season'].astype(str) + df['DayNum'].astype(str) +
                         df['WTeamID'].astype(str) + df['LTeamID'].astype(str))

        # データをTeamIDに対するテーブルに変換する
        wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
        lcols = [c for c in df.columns if 'L' in c and c != 'WLoc']

        t1df = df[['MatchID', 'Season', 'DayNum'] + wcols + ['LScore', 'LOR', 'LDR']].copy()
        rename_dict = dict(zip(wcols, [utils.lreplace('W', '', c) for c in wcols]))
        rename_dict['LScore'] = 'OppScore'
        rename_dict['LOR'] = 'OppOR'
        rename_dict['LDR'] = 'OppDR'
        t1df.rename(columns=rename_dict, inplace=True)
        t1df['Result'] = 1
        t1df['IsHome'] = (df['WLoc'] == 'H')

        t2df = df[['MatchID', 'Season', 'DayNum'] + lcols + ['WScore', 'WOR', 'WDR']].copy()
        rename_dict = dict(zip(lcols, [utils.lreplace('L', '', c) for c in lcols]))
        rename_dict['WScore'] = 'OppScore'
        rename_dict['WOR'] = 'OppOR'
        rename_dict['WDR'] = 'OppDR'
        t2df.rename(columns=rename_dict, inplace=True)
        t2df['Result'] = 0
        t2df['IsHome'] = (df['WLoc'] == 'A')

        df = pd.concat([t1df, t2df], axis=0)

        # Possessions = 0.96 x (FGA + Turnovers + (0.475 x FTA) - Offensive Rebounds
        df['Poss'] = 0.96 * (df['FGA'] + df['TO'] + ((0.475) * df['FTA']) - df['OR'])
        # Offensive Rating = 100 x (Score / Possessions)
        df['OffRtg'] = 100 * (df['Score'] / df['Poss'])
        # Defensive Rating = 100 x (Opponent's Score / Possessions)
        df['DefRtg'] = 100 * (df['OppScore'] / df['Poss'])
        # Net Rating = 100 x (Offensive Rating - Defensive Rating)
        df['NetRtg'] = 100 * (df['OffRtg'] - df['DefRtg'])
        # Assist ratio = 100 * Assists / (FGA + (0.475 * FTA) + Asistst + Turnovers))
        df['AstRto'] = 100 * df['Ast'] / (df['FGA'] + (0.475 * df['FTA']) + df['Ast'] + df['TO'])
        # Turnover ratio = 100 * Turnovers / (FGA + (0.475 * FTA) + Assists + Turnovers)
        df['TORto'] = 100 * df['TO'] / (df['FGA'] + (0.475 * df['FTA']) + df['Ast'] + df['TO'])
        # True Shooting % = 100 * Team Points / (2 * (FGA + (0.475 * FTA)))
        df['TruShtPct'] = 100 * df['Score'] / (2 * (df['FGA'] + (0.475 * df['FTA'])))
        # Effective FG% = (FGM + 0.5 * Threes Made) / FGA
        df['EffFGPct'] = (df['FGM'] + 0.5 * df['FGM3']) / df['FGA']
        # Free Throw rate = FTA / FGA
        df['FTR'] = df['FTA'] / df['FGA']
        # Offensive Rebound % = Offensive Rebounds / (Offensive Rebounds + Opponent's Defensive Rebounds)
        df['ORPct'] = df['OR'] / (df['OR'] + df['OppOR'])
        # Defensive Rebound % = Defensive Rebounds / (Defensive Rebounds + Opponent's Offensive Rebounds)
        df['DRPct'] = df['DR'] / (df['DR'] + df['OppDR'])
        # Total Rebound % = (Defensive Rebounds + Offensive Rebounds) /
        #                   (Defensive Rebounds + Offensive Rebounds
        #                   + Opponent's Defensive Rebounds + Opponent's Offensive Rebounds)
        df['TRPct'] = (df['DR'] + df['OR']) / (df['DR'] + df['OR'] + df['OppDR'] + df['OppOR'])

        feat = df.groupby(['Season', 'TeamID']).agg({
            'Poss': ['mean', 'median', 'min', 'max'],
            'OffRtg': ['mean', 'median', 'min', 'max'],
            'DefRtg': ['mean', 'median', 'min', 'max'],
            'NetRtg': ['mean', 'median', 'min', 'max'],
            'AstRto': ['mean', 'median', 'min', 'max'],
            'TORto': ['mean', 'median', 'min', 'max'],
            'TruShtPct': ['mean', 'median', 'min', 'max'],
            'EffFGPct': ['mean', 'median', 'min', 'max'],
            'FTR': ['mean', 'median', 'min', 'max'],
            'ORPct': ['mean', 'median', 'min', 'max'],
            'DRPct': ['mean', 'median', 'min', 'max'],
        })

        advanced_stats_cols = ['Poss', 'OffRtg', 'DefRtg', 'NetRtg', 'AstRto', 'TORto', 'TruShtPct',
                               'EffFGPct', 'FTR', 'ORPct', 'DRPct']
        feat.columns = ["_".join(x) for x in feat.columns.ravel()]
        feat = pd.concat([
            feat,
            df[df.DayNum > 118].groupby(['Season', 'TeamID'])[advanced_stats_cols].mean().rename(
                columns=dict(
                    zip(advanced_stats_cols, [c + '14D_mean' for c in advanced_stats_cols]))
            )], axis=1).fillna(0).reset_index()
        return feat

    def post_process(self, trn, tst):
        return trn, tst


if __name__ == '__main__':
    train, test = _103_SeasonAdvancedStats().create_feature(devmode=True)
