"""Touney Stats History
"""
import os
import pandas as pd

import CONST
from feature import FeatureBase

import feature._001_utils as utils


class _302_TourneyHistoryStats(FeatureBase):
    fin = os.path.join(CONST.INDIR, 'NCAATourneyDetailedResults.csv')

    def create_feature_impl(self, df):
        df['Season'] = df['Season'] - 1
        uni_season = df.Season.unique()
        feat = pd.DataFrame()

        for s in uni_season:
            tidy_df = tidy_tourney_detailed_data(df[df['Season'] <= s].copy())

            # Possessions = 0.96 x (FGA + Turnovers + (0.475 x FTA) - Offensive Rebounds
            tidy_df['Poss'] = 0.96 * (tidy_df['FGA'] + tidy_df['TO'] + ((0.475) * tidy_df['FTA']) - tidy_df['OR'])
            # Offensive Rating = 100 x (Score / Possessions)
            tidy_df['OffRtg'] = 100 * (tidy_df['Score'] / tidy_df['Poss'])
            # Defensive Rating = 100 x (Opponent's Score / Possessions)
            tidy_df['DefRtg'] = 100 * (tidy_df['OppScore'] / tidy_df['Poss'])
            # Net Rating = 100 x (Offensive Rating - Defensive Rating)
            tidy_df['NetRtg'] = 100 * (tidy_df['OffRtg'] - tidy_df['DefRtg'])
            # Assist ratio = 100 * Assists / (FGA + (0.475 * FTA) + Asistst + Turnovers))
            tidy_df['AstRto'] = 100 * tidy_df['Ast'] / (
                    tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO'])
            # Turnover ratio = 100 * Turnovers / (FGA + (0.475 * FTA) + Assists + Turnovers)
            tidy_df['TORto'] = 100 * tidy_df['TO'] / (
                    tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO'])
            # True Shooting % = 100 * Team Points / (2 * (FGA + (0.475 * FTA)))
            tidy_df['TruShtPct'] = 100 * tidy_df['Score'] / (2 * (tidy_df['FGA'] + (0.475 * tidy_df['FTA'])))
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
            tidy_df['TRPct'] = (tidy_df['DR'] + tidy_df['OR']) / (
                    tidy_df['DR'] + tidy_df['OR'] + tidy_df['OppDR'] + tidy_df['OppOR'])

            tmp = tidy_df.groupby(['Season', 'TeamID']).agg({
                'Score': ['mean', 'median'],
                'OppScore': ['mean'],
                'FGA': ['mean', 'median', 'min', 'max'],
                'Ast': ['mean'],
                'Blk': ['mean'],
                'OppFGA': ['mean', 'min'],
                'Poss': ['mean', 'median'],
                'OffRtg': ['mean'],
                'DefRtg': ['mean'],
                'NetRtg': ['mean'],
                'AstRto': ['mean'],
                'TORto': ['mean'],
                'TruShtPct': ['mean'],
                'EffFGPct': ['mean'],
                'FTR': ['mean'],
                'ORPct': ['mean'],
                'DRPct': ['mean'],
            })

            tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]
            tmp = tmp.reset_index()
            tmp['Season'] = s + 1
            feat = pd.concat([feat, tmp], axis=0)

        return feat

    def post_process(self, trn, tst):
        return trn.fillna(0), tst.fillna(0)


def tidy_tourney_detailed_data(df):
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

    return df


if __name__ == '__main__':
    train, test = _302_TourneyHistoryStats().create_feature(devmode=True)
    #
    # df = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyDetailedResults.csv'))
    # df['Season'] = df['Season'] - 1
    # uni_season = df.Season.unique()
    # feat = pd.DataFrame()
    # for s in uni_season:
    #     tidy_df = tidy_tourney_detailed_data(df[df['Season'] <= s].copy())
    #
    #     # Possessions = 0.96 x (FGA + Turnovers + (0.475 x FTA) - Offensive Rebounds
    #     tidy_df['Poss'] = 0.96 * (tidy_df['FGA'] + tidy_df['TO'] + ((0.475) * tidy_df['FTA']) - tidy_df['OR'])
    #     # Offensive Rating = 100 x (Score / Possessions)
    #     tidy_df['OffRtg'] = 100 * (tidy_df['Score'] / tidy_df['Poss'])
    #     # Defensive Rating = 100 x (Opponent's Score / Possessions)
    #     tidy_df['DefRtg'] = 100 * (tidy_df['OppScore'] / tidy_df['Poss'])
    #     # Net Rating = 100 x (Offensive Rating - Defensive Rating)
    #     tidy_df['NetRtg'] = 100 * (tidy_df['OffRtg'] - tidy_df['DefRtg'])
    #     # Assist ratio = 100 * Assists / (FGA + (0.475 * FTA) + Asistst + Turnovers))
    #     tidy_df['AstRto'] = 100 * tidy_df['Ast'] / (
    #             tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO'])
    #     # Turnover ratio = 100 * Turnovers / (FGA + (0.475 * FTA) + Assists + Turnovers)
    #     tidy_df['TORto'] = 100 * tidy_df['TO'] / (
    #             tidy_df['FGA'] + (0.475 * tidy_df['FTA']) + tidy_df['Ast'] + tidy_df['TO'])
    #     # True Shooting % = 100 * Team Points / (2 * (FGA + (0.475 * FTA)))
    #     tidy_df['TruShtPct'] = 100 * tidy_df['Score'] / (2 * (tidy_df['FGA'] + (0.475 * tidy_df['FTA'])))
    #     # Effective FG% = (FGM + 0.5 * Threes Made) / FGA
    #     tidy_df['EffFGPct'] = (tidy_df['FGM'] + 0.5 * tidy_df['FGM3']) / tidy_df['FGA']
    #     # Free Throw rate = FTA / FGA
    #     tidy_df['FTR'] = tidy_df['FTA'] / tidy_df['FGA']
    #     # Offensive Rebound % = Offensive Rebounds / (Offensive Rebounds + Opponent's Defensive Rebounds)
    #     tidy_df['ORPct'] = tidy_df['OR'] / (tidy_df['OR'] + tidy_df['OppOR'])
    #     # Defensive Rebound % = Defensive Rebounds / (Defensive Rebounds + Opponent's Offensive Rebounds)
    #     tidy_df['DRPct'] = tidy_df['DR'] / (tidy_df['DR'] + tidy_df['OppDR'])
    #     # Total Rebound % = (Defensive Rebounds + Offensive Rebounds) /
    #     #                   (Defensive Rebounds + Offensive Rebounds
    #     #                   + Opponent's Defensive Rebounds + Opponent's Offensive Rebounds)
    #     tidy_df['TRPct'] = (tidy_df['DR'] + tidy_df['OR']) / (
    #             tidy_df['DR'] + tidy_df['OR'] + tidy_df['OppDR'] + tidy_df['OppOR'])
    #
    #     tmp = tidy_df.groupby(['Season', 'TeamID']).agg({
    #         'Score': ['mean', 'median'],
    #         'OppScore': ['mean'],
    #         'FGA': ['mean', 'median', 'min', 'max'],
    #         'Ast': ['mean'],
    #         'Blk': ['mean'],
    #         'OppFGA': ['mean', 'min'],
    #         'Poss': ['mean', 'median'],
    #         'OffRtg': ['mean'],
    #         'DefRtg': ['mean'],
    #         'NetRtg': ['mean'],
    #         'AstRto': ['mean'],
    #         'TORto': ['mean'],
    #         'TruShtPct': ['mean'],
    #         'EffFGPct': ['mean'],
    #         'FTR': ['mean'],
    #         'ORPct': ['mean'],
    #         'DRPct': ['mean'],
    #     })
    #
    #     advanced_stats_cols = ['Poss', 'OffRtg', 'DefRtg', 'NetRtg', 'AstRto', 'TORto', 'TruShtPct',
    #                            'EffFGPct', 'FTR', 'ORPct', 'DRPct']
    #     tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]
    #     tmp = tmp.reset_index()
    #
    #     feat = pd.concat([feat, tmp], axis=0)
