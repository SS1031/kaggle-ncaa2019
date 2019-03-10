"""https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import CONST

K = 20.
HOME_ADVANTAGE = 100.

rscr = pd.read_csv(os.path.join(CONST.INDIR, "RegularSeasonCompactResults.csv"))
rscr.head(3)

team_ids = set(rscr.WTeamID).union(set(rscr.LTeamID))
print(len(team_ids))

# This dictionary will be used as a lookup for current
# scores while the algorithm is iterating through each game
elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

# Elo updates will be scaled based on the margin of victory
rscr['margin'] = rscr.WScore - rscr.LScore


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
assert np.all(rscr.index.values == np.array(range(rscr.shape[0]))), "Index is out of order."
preds = []
w_elo = []
l_elo = []

# Loop over all rows of the games dataframe
for row in rscr.itertuples():
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

rscr['w_elo'] = w_elo
rscr['l_elo'] = l_elo

print("ELO ratiings predictions ", np.mean(-np.log(preds)))


def final_elo_per_season(df, team_id):
    d = df.copy()
    d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]
    d.sort_values(['Season', 'DayNum'], inplace=True)
    d.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = d.WTeamID == team_id
    l_mask = d.LTeamID == team_id
    d['season_elo'] = None
    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
    out = pd.DataFrame({
        'TeamID': team_id,
        'Season': d.Season,
        'SeasonELO': d.season_elo
    }).reset_index(drop=True)

    return out


df_list = [final_elo_per_season(rscr, id) for id in team_ids]
season_elos = pd.concat(df_list)
