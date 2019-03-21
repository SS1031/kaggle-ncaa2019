import os
import re
import numpy as np
import pandas as pd
import CONST


###################################################################################################
#
# https://www.kaggle.com/fabiendaniel/elo-world
#
###################################################################################################
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        print("******************************")
        print("Column: ", col)
        print("dtype before: ", col_type)
        if col_type != object and col_type != 'datetime64[ns]':  # Exclude strings and datetime
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        # Print new column type
        print("dtype after: ", df[col].dtype)
        print("******************************")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    df.info()

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
        )
    return df


def to_feather(df, save_dir):
    if df.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {df.columns[df.columns.duplicated()]}')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(os.path.join(save_dir, f'{c}.f'))
    return


def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)


def load_trn_base():
    c_tourney = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyCompactResults.csv'))
    d_season = pd.read_csv(os.path.join(CONST.INDIR, "RegularSeasonDetailedResults.csv"))
    c_tourney = c_tourney[c_tourney.Season > d_season.Season.min()].reset_index(drop=True)

    # データをT1, T2チームが両方勝つように二倍にする
    d_tour_wcols = [c for c in c_tourney.columns if 'W' in c and c != 'WLoc']
    d_tour_lcols = [c for c in c_tourney.columns if 'L' in c]

    # WTeamID -> T1TeamID, LTeamID -> T2TeamIDへ変換
    rename_dict = dict(zip(d_tour_wcols, [lreplace('W', 'T1', c) for c in d_tour_wcols]))
    rename_dict.update(dict(zip(d_tour_lcols, [lreplace('L', 'T2', c) for c in d_tour_lcols])))
    rename_dict.update({'WLoc': 'T1Home'})
    c_tourney1 = c_tourney.copy().rename(columns=rename_dict)
    c_tourney1['Result'] = 1
    c_tourney1['T1Home'] = (c_tourney1['T1Home'] == 'H')

    # LTeamID -> T1TeamID, WTeamID -> T2TeamIDへ変換
    rename_dict = dict(zip(d_tour_lcols, [lreplace('L', 'T1', c) for c in d_tour_lcols]))
    rename_dict.update(dict(zip(d_tour_wcols, [lreplace('W', 'T2', c) for c in d_tour_wcols])))
    rename_dict.update({'WLoc': 'T1Home'})
    c_tourney2 = c_tourney.copy().rename(columns=rename_dict)
    c_tourney2['T1Home'] = (c_tourney2['T1Home'] == 'A')
    c_tourney2['Result'] = 0

    # 変換したのをくっつける
    trn_base = pd.concat([c_tourney1, c_tourney2], axis=0)[
        ['Season', 'T1TeamID', 'T2TeamID', 'Result']
    ].reset_index(drop=True)

    return trn_base


def load_tst_base():
    ss = pd.read_csv(CONST.SS)
    tst = ss.ID.str.split('_', expand=True).astype(int)
    tst.columns = ['Season', 'T1TeamID', 'T2TeamID']

    return tst


def tidy_detailed_data(df):
    initial_nrow = df.shape[0]
    df['MatchID'] = (df['Season'].astype(str) + df['DayNum'].astype(str) +
                     df['WTeamID'].astype(str) + df['LTeamID'].astype(str))

    # データをTeamIDに対
    # るテーブルに変換する
    wcols = [c for c in df.columns if 'W' in c and c != 'WLoc']
    lcols = [c for c in df.columns if 'L' in c and c != 'WLoc']

    t1df = df[['MatchID', 'Season', 'DayNum'] + wcols + lcols].copy()
    rename_dict = dict(zip(wcols, [lreplace('W', '', c) for c in wcols]))
    rename_dict.update(
        dict(zip(lcols, [lreplace('L', 'Opp', c) for c in lcols]))
    )
    t1df.rename(columns=rename_dict, inplace=True)

    t1df['IsHome'] = (df.WLoc == 'H')
    t1df['Result'] = 1

    t2df = df[['MatchID', 'Season', 'DayNum'] + lcols + wcols].copy()
    rename_dict = dict(zip(lcols, [lreplace('L', '', c) for c in lcols]))
    rename_dict.update(
        dict(zip(wcols, [lreplace('W', 'Opp', c) for c in wcols]))
    )
    t2df.rename(columns=rename_dict, inplace=True)
    t2df['IsHome'] = (df.WLoc == 'A')
    t2df['Result'] = 0

    df = pd.concat([t1df, t2df], axis=0).reset_index(drop=True)

    assert (initial_nrow * 2) == df.shape[0]

    return df


if __name__ == '__main__':
    trn = load_trn_base()
    print(trn.head())
