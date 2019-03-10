# Revision History
# Version 7: Fixed regular season stat bug
# Version 6: Added submission code

# This kernel creates basic logistic regression models and provides a
# mechanism to select attributes and check results against tournaments since 2013

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import CONST

tourney_cresults = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyCompactResults.csv'))
tourney_cresults = tourney_cresults.loc[tourney_cresults['Season'] >= 2003]

training_set = pd.DataFrame()
training_set['Result'] = np.random.randint(0, 2, len(tourney_cresults.index))
training_set['Season'] = tourney_cresults['Season'].values
training_set['Team1'] = (training_set['Result'].values * tourney_cresults['WTeamID'].values +
                         (1 - training_set['Result'].values) * tourney_cresults['LTeamID'].values)
training_set['Team2'] = ((1 - training_set['Result'].values) * tourney_cresults['WTeamID'].values +
                         training_set['Result'].values * tourney_cresults['LTeamID'].values)

# Calculate Delta Seeds
seeds = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
seeds['Seed'] = pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer', errors='coerce')


def delta_seed(row):
    cond = (seeds['Season'] == row['Season'])
    return (seeds[cond & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] -
            seeds[cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0])


training_set['deltaSeed'] = training_set.apply(delta_seed, axis=1)

# Calculate Delta Ordinals
mo = pd.read_csv(os.path.join(CONST.INDIR, 'MasseyOrdinals.csv'))
mo = mo[(mo['RankingDayNum'] == 128) & (mo['Season'] >= 2003)]  # See Note on MO


def delta_ord(row):
    cond = (mo['Season'] == row['Season'])
    cond1 = (mo['TeamID'] == row['Team1']) & cond
    cond2 = (mo['TeamID'] == row['Team2']) & cond
    t1 = mo[cond1]['OrdinalRank'].mean()
    t2 = mo[cond2]['OrdinalRank'].mean()
    return t1 - t2


training_set['deltaMO'] = training_set.apply(delta_ord, axis=1)

# Calculate win pct
season_dresults = pd.read_csv(os.path.join(CONST.INDIR, 'RegularSeasonDetailedResults.csv'))
record = pd.DataFrame({
    'wins': season_dresults.groupby(['Season', 'WTeamID']).size()
}).reset_index()
losses = pd.DataFrame({
    'losses': season_dresults.groupby(['Season', 'LTeamID']).size()}
).reset_index()

record = record.merge(losses, how='outer',
                      left_on=['Season', 'WTeamID'],
                      right_on=['Season', 'LTeamID'])
record = record.fillna(0)
record['games'] = record['wins'] + record['losses']


def delta_winPct(row):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])
    return (record[cond1]['wins'] / record[cond1]['games']).mean() - (
            record[cond2]['wins'] / record[cond2]['games']).mean()


training_set['deltaWinPct'] = training_set.apply(delta_winPct, axis=1)

dfW = season_dresults.groupby(['Season', 'WTeamID']).sum().reset_index()
dfL = season_dresults.groupby(['Season', 'LTeamID']).sum().reset_index()


def get_points_for(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID'])
    fld1 = 'WScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID'])
    fld2 = 'LScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


def get_points_against(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID'])
    fld1 = 'LScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID'])
    fld2 = 'WScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


record['PointsFor'] = record.apply(get_points_for, axis=1)
record['PointsAgainst'] = record.apply(get_points_against, axis=1)


def get_remaining_stats(row, field):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID'])
    fld1 = 'W' + field
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID'])
    fld2 = 'L' + field
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal


cols = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

for col in cols:
    print("Processing", col)
    record[col] = record.apply(get_remaining_stats, args=(col,), axis=1)


def delta_stat(row, field):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])
    return (record[cond1][field] / record[cond1]['games']).mean() - (
            record[cond2][field] / record[cond2]['games']).mean()


cols = ['PointsFor', 'PointsAgainst', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM',
        'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

for col in cols:
    print("Processing", col)
    training_set['delta' + col] = training_set.apply(delta_stat, args=(col,), axis=1)

training_set.describe()

# Train a model on all of the data
import statsmodels.api as sm

# Field descriptions:
# deltaSeed: difference in team's seeds
# deltaMO: difference in team's Massey Ordinals on day 128
# deltaWinPct: difference in the team's winning percentage
# deltaPointsFor: difference in the average points scored per game
# deltaPointsAgainst: difference in the average points scored agains the teams
# deltaFGM: difference in the field goals made per game
# deltaFGA: difference in the field goals attempted per game
# deltaFGM3: difference in 3 point fields goals made per game
# deltaFGA3: difference in the 3 points fields goals attempted per game
# deltaFTM: difference in free throws made per game
# deltaFTA: difference in free throws attempted per game
# deltaOR: difference in offence rebounds per game
# deltaDR: difference in defensive rebounds per game
# deltaAst: difference in assists per game
# deltaTO: difference in turnovers per game
# deltaStl: difference in steals per game
# deltaBlk: difference in blocks per game
# deltaPF: difference in personal fouls per game

# You would probabaly want to select a subset of these attributes
cols = ['deltaSeed', 'deltaMO', 'deltaWinPct', 'deltaPointsFor', 'deltaPointsAgainst',
        'deltaFGM', 'deltaFGA', 'deltaFGM3', 'deltaFGA3', 'deltaFTM', 'deltaFTA',
        'deltaOR', 'deltaDR', 'deltaAst', 'deltaTO', 'deltaStl', 'deltaBlk', 'deltaPF']
X = training_set[cols]
y = training_set['Result']

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())
