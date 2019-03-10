"""A Very Extensive NCAA Exploratory Analysis
https://www.kaggle.com/captcalculator/a-very-extensive-ncaa-exploratory-analysis
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import CONST

plt.style.use('seaborn')

# Data Section 1
df_teams = pd.read_csv(os.path.join(CONST.INDIR, 'Teams.csv'))
df_seasons = pd.read_csv(os.path.join(CONST.INDIR, 'Seasons.csv'))
df_seeds = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'))
df_seas_results = pd.read_csv(os.path.join(CONST.INDIR, 'RegularSeasonCompactResults.csv'))
df_tour_results = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyCompactResults.csv'))
df_seas_detail = pd.read_csv(os.path.join(CONST.INDIR, 'RegularSeasonDetailedResults.csv'))
df_tour_detail = pd.read_csv(os.path.join(CONST.INDIR, 'NCAATourneyDetailedResults.csv'))
df_conferences = pd.read_csv(os.path.join(CONST.INDIR, 'Conferences.csv'))
df_team_conferences = pd.read_csv(os.path.join(CONST.INDIR, 'TeamConferences.csv'))
df_coaches = pd.read_csv(os.path.join(CONST.INDIR, 'TeamCoaches.csv'))

print("### Teams.csv")
print(df_teams.describe(include='all'))
print("\n\n### Seasons.csv")
print(df_seasons.describe(include='all'))
print("\n\n### NCAATourneySeeds.csv")
print(df_seeds.describe(include='all'))
print("\n\n### RegularSeasonCompactResults.csv")
print(df_seas_results.describe(include='all'))
print("\n\n### NCAATourneyCompactResults.csv")
print(df_tour_results.describe(include='all'))
print("\n\n### RegularSeasonDetailedResults.csv")
print(df_seas_detail.describe(include='all'))
print("\n\n### NCAATourneyDetailedResults.csv")
print(df_tour_detail.describe(include='all'))
print("\n\n### Conferences.csv")
print(df_conferences.describe(include='all'))
print("\n\n### TeamsConferences.csv")
print(df_team_conferences.describe(include='all'))
print("\n\n### TeamCoaches.csv")
print(df_coaches.describe(include='all'))

# Historical Performance
print("### No.1 Seed Since 1985.")
tmp = df_seeds.merge(
    df_teams[['TeamID', 'TeamName']], on='TeamID', how='left'
)
tmp['Seed'] = tmp['Seed'].str[1:3].astype(int)
print(tmp.groupby(['Seed', 'TeamName']).size()[1].sort_values(ascending=False)[0:15])

print("### Regular Seasons Wins Since 1985.")
tmp = df_seas_results.merge(
    df_teams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left'
)
print(tmp.groupby(['TeamName']).size().sort_values(ascending=False)[0:15])

print("### Tournament Wins Since 1985.")
tmp = df_tour_results.merge(
    df_teams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left'
)
print(tmp.groupby(['TeamName']).size().sort_values(ascending=False)[0:15])

print("### Tournament Championships Since 1985. (The championship game of the men's tournament is on DayNum=154)")
tmp = df_tour_results[df_tour_results.DayNum == 154].merge(
    df_teams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left'
)
print(tmp.groupby(['TeamName']).size().sort_values(ascending=False)[0:15])

# Conferences
print("### Tournament Wins by Conferences Since 1985.")
print(df_tour_results.merge(
    df_team_conferences.merge(df_conferences, on='ConfAbbrev')[['TeamID', 'Description']],
    left_on='WTeamID', right_on='TeamID', how='left'
).groupby('Description').size().sort_values(ascending=False)[0:15])
print("### Championships by Conferences Since 1985.")
print(df_tour_results[df_tour_results.DayNum == 154].merge(
    df_team_conferences.merge(df_conferences, on='ConfAbbrev')[['TeamID', 'Description']],
    left_on='WTeamID', right_on='TeamID', how='left'
).groupby('Description').size().sort_values(ascending=False)[0:15])

# Notes: Team Conference will be changed for each seasons
top_conf = ['acc', 'big_east', 'sec', 'big_ten', 'pac_ten', 'big_twelve']
tmp = df_tour_results.merge(
    pd.concat((df_team_conferences['Season'], df_team_conferences[['TeamID', 'ConfAbbrev']].add_prefix('W')), axis=1),
    on=['Season', 'WTeamID'], how='left').merge(
    pd.concat((df_team_conferences['Season'], df_team_conferences[['TeamID', 'ConfAbbrev']].add_prefix('L')), axis=1),
    on=['Season', 'LTeamID'], how='left'
)
print(tmp[tmp.WConfAbbrev.isin(top_conf) & tmp.LConfAbbrev.isin(top_conf)].groupby(
    ['WConfAbbrev', 'LConfAbbrev']
).size().unstack(fill_value=0))


# Indicators of Regular Season Success
def add_stats_columns(df):
    df['FGP'] = df['FGM'] / df['FGA']
    df['FGP2'] = (df['FGM'] - df['FGM3']) / (df['FGA'] - df['FGA3'])
    df['FGP3'] = df['FGM3'] / df['FGA3']
    df['FTP'] = df['FTM'] / df['FTA']
    return df


win_cols = [c for c in df_seas_detail.columns if c[0] == "W"]
df_win_season_detail = df_seas_detail[['Season'] + win_cols].rename(
    columns=dict(zip(win_cols, [c[1:] for c in win_cols]))
)
df_win_season_detail = add_stats_columns(df_win_season_detail)

los_cols = [c for c in df_seas_detail.columns if c[0] == "L"]
df_los_season_detail = df_seas_detail[['Season'] + los_cols].rename(
    columns=dict(zip(los_cols, [c[1:] for c in los_cols]))
)
df_win_season_detail = add_stats_columns(df_win_season_detail)
df_los_season_detail = add_stats_columns(df_los_season_detail)

# Interaction Stats
df_win_season_detail['ORP'] = df_win_season_detail['OR'] / (df_win_season_detail['OR'] + df_los_season_detail['DR'])
df_los_season_detail['ORP'] = df_los_season_detail['OR'] / (df_win_season_detail['OR'] + df_los_season_detail['DR'])
df_win_season_detail['DRP'] = df_win_season_detail['DR'] / (df_win_season_detail['OR'] + df_los_season_detail['DR'])
df_los_season_detail['DRP'] = df_los_season_detail['DR'] / (df_win_season_detail['OR'] + df_los_season_detail['DR'])

for c in ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM',
          'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'FGP', 'FGP2',
          'FGP3', 'FTP', 'ORP', 'DRP']:
    sns.distplot(df_win_season_detail[[c]], hist=False, rug=False, label='Win')
    sns.distplot(df_los_season_detail[[c]], hist=False, rug=False, label='Lose')
    plt.title(c)
    plt.legend()
    plt.show()

# Predictors of Tournament Success
tmp = pd.concat([df_seas_results.groupby(['Season', 'WTeamID']).size().to_frame('WinSeason'),
                 df_tour_results.groupby(['Season', 'WTeamID']).size().to_frame('WinTourney')],
                axis=1).dropna().reset_index()

# Show the results of a linear regression within each dataset
g = sns.lmplot(x='WinSeason', y='WinTourney', col="Season", data=tmp, col_wrap=6)
g.set(xlim=(tmp.WinSeason.min() - 1, tmp.WinSeason.max() + 1),
      ylim=(tmp.WinTourney.min() - 1, tmp.WinTourney.max() + 1))
g.fig.set_figheight(10)
g.fig.set_figwidth(10)
g.fig.suptitle('Tournament Wins by Regular Season Scores per Game')
plt.show()

# Tournament Wins by Regular Season Scores per Game
win = df_seas_results.groupby(['Season', 'WTeamID']).agg({'WTeamID': 'size', 'WScore': 'sum'})
win.index.names = ['Season', 'TeamID']
win.columns = ['WMatches', 'WTotalScore']
los = df_seas_results.groupby(['Season', 'LTeamID']).agg({'LTeamID': 'size', 'LScore': 'sum'})
los.columns = ['LMatches', 'LTotalScore']
los.index.names = ['Season', 'TeamID']
tmp = pd.concat([win, los], axis=1)

tmp['Matches'] = tmp['WMatches'] + tmp['LMatches']
tmp['TotalScore'] = tmp['WTotalScore'] + tmp['LTotalScore']
tmp['AVGSeasonScore'] = tmp['TotalScore'] / tmp['Matches']

win_tour = df_tour_results.groupby(['Season', 'WTeamID']).size().to_frame('WinTourney')
win_tour.index.names = ['Season', 'TeamID']

tmp = pd.concat([tmp[['AVGSeasonScore']], win_tour], axis=1).dropna().reset_index()
g = sns.lmplot(x='AVGSeasonScore', y='WinTourney', col="Season", data=tmp, col_wrap=6)
g.set(xlim=(tmp.AVGSeasonScore.min() - 1, tmp.AVGSeasonScore.max() + 1),
      ylim=(tmp.WinTourney.min() - 1, tmp.WinTourney.max() + 1))
g.fig.set_figheight(10)
g.fig.set_figwidth(10)
g.fig.suptitle('Tournament Wins by Regular Season Scores per Game')
plt.tight_layout()
plt.show()

tmp = df_tour_results[['Season', 'WTeamID']].rename(columns={'WTeamID': 'TeamID'}).merge(
    df_seeds, on=['Season', 'TeamID'])
tmp['Seed'] = tmp.Seed.str[1:3].astype(int)
tmp = pd.concat([tmp.groupby(['Season', 'TeamID']).TeamID.size(),
                 tmp.groupby(['Season', 'TeamID']).Seed.unique().apply(lambda x: x[0])], axis=1)
tmp.columns = ['WinTourney', 'Seed']
tmp.reset_index(inplace=True)
g = sns.lmplot(x='Seed', y='WinTourney', col="Season", data=tmp, col_wrap=6)
g.set(xlim=(tmp.Seed.min() - 1, tmp.Seed.max() + 1),
      ylim=(tmp.WinTourney.min() - 1, tmp.WinTourney.max() + 1))
g.fig.set_figheight(10)
g.fig.set_figwidth(10)
g.fig.suptitle('Tournament Wins by Seed')
plt.tight_layout()
plt.show()

# Look at the percentage of times the better-seeded team won by season.
tmp = df_tour_results.merge(
    df_seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
    on=['Season', 'WTeamID'], how='left'
).merge(
    df_seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
    on=['Season', 'LTeamID'], how='left'
)
tmp['WSeed'] = tmp['WSeed'].str[1:3].astype(int)
tmp['LSeed'] = tmp['LSeed'].str[1:3].astype(int)
tmp['WinBetterSeed'] = tmp['WSeed'] < tmp['LSeed']
tmp.groupby('Season').WinBetterSeed.mean().sort_values().plot(kind='bar')
