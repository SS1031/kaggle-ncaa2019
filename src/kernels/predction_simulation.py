import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

prediction_file = '../data/output/sbmt/config006.csv'

# BASED ON ESPN.COM
round_scores = {
    1: 10,
    2: 20,
    3: 40,
    4: 80,
    5: 160,
    6: 360
}
region_pairings = (('east', 'west'), ('midwest', 'south'))
seed_pairs_by_round = {
    1: {
        1: 16,
        8: 9,
        5: 12,
        4: 13,
        6: 11,
        3: 14,
        7: 10,
        2: 15,
    },
    2: {
        1: 8,
        4: 5,
        3: 6,
        2: 7,
    },
    3: {
        1: 4,
        2: 3,
    },
    4: {
        1: 2,
    },
}
team1 = []
team2 = []
prob = []
delim = 'beats'
for line in open(prediction_file, 'r').readlines():
    team1.append(line.split(delim)[0])
    team2.append(line.split(delim)[1].split(':')[0])
    prob.append(float(line.strip().split(':')[1]))

team1 = np.array(team1)
team2 = np.array(team2)
probs = np.array(prob)

submission_dict = {}  # this will store the probabilites from Finalpredictions.csv
for team1i in team1:
    submission_dict[team1i.strip()] = {}
for team1i, team2i, prob in zip(team1, team2, probs):
    submission_dict[team1i.strip()][team2i.strip()] = prob


def get_starting_bracket():
    # Makes an empty bracket based on 2018 Tournament. Will be updated for 2019 when available.
    starting_bracket = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 'Final': [], 'Winner': None}
    for rnd in [1, 2, 3, 4, 5]:
        starting_bracket[rnd] = {'east': {}, 'west': {}, 'midwest': {}, 'south': {}}

    rnd = 1
    loc = 'south'
    starting_bracket[rnd][loc][1] = 'Virginia'
    starting_bracket[rnd][loc][2] = 'Cincinnati'
    starting_bracket[rnd][loc][3] = 'Tennessee'
    starting_bracket[rnd][loc][4] = 'Arizona'
    starting_bracket[rnd][loc][5] = 'Kentucky'
    starting_bracket[rnd][loc][6] = 'Miami FL'
    starting_bracket[rnd][loc][7] = 'Nevada'
    starting_bracket[rnd][loc][8] = 'Creighton'
    starting_bracket[rnd][loc][9] = 'Kansas St'
    starting_bracket[rnd][loc][10] = 'Texas'
    starting_bracket[rnd][loc][11] = 'Loyola-Chicago'
    starting_bracket[rnd][loc][12] = 'Davidson'
    starting_bracket[rnd][loc][13] = 'Buffalo'
    starting_bracket[rnd][loc][14] = 'Wright St'
    starting_bracket[rnd][loc][15] = 'Georgia St'
    starting_bracket[rnd][loc][16] = 'UMBC'

    loc = 'west'
    starting_bracket[rnd][loc][1] = 'Xavier'
    starting_bracket[rnd][loc][2] = 'North Carolina'
    starting_bracket[rnd][loc][3] = 'Michigan'
    starting_bracket[rnd][loc][4] = 'Gonzaga'
    starting_bracket[rnd][loc][5] = 'Ohio St'
    starting_bracket[rnd][loc][6] = 'Houston'
    starting_bracket[rnd][loc][7] = 'Texas A&M'
    starting_bracket[rnd][loc][8] = 'Missouri'
    starting_bracket[rnd][loc][9] = 'Florida St'
    starting_bracket[rnd][loc][10] = 'Providence'
    starting_bracket[rnd][loc][11] = 'San Diego St'
    starting_bracket[rnd][loc][12] = 'S Dakota St'
    starting_bracket[rnd][loc][13] = 'UNC Greensboro'
    starting_bracket[rnd][loc][14] = 'Montana'
    starting_bracket[rnd][loc][15] = 'Lipscomb'
    starting_bracket[rnd][loc][16] = 'TX Southern'

    loc = 'midwest'
    starting_bracket[rnd][loc][1] = 'Kansas'
    starting_bracket[rnd][loc][2] = 'Duke'
    starting_bracket[rnd][loc][3] = 'Michigan St'
    starting_bracket[rnd][loc][4] = 'Auburn'
    starting_bracket[rnd][loc][5] = 'Clemson'
    starting_bracket[rnd][loc][6] = 'TCU'
    starting_bracket[rnd][loc][7] = 'Rhode Island'
    starting_bracket[rnd][loc][8] = 'Seton Hall'
    starting_bracket[rnd][loc][9] = 'NC State'
    starting_bracket[rnd][loc][10] = 'Oklahoma'
    starting_bracket[rnd][loc][11] = 'Syracuse'
    starting_bracket[rnd][loc][12] = 'New Mexico St'
    starting_bracket[rnd][loc][13] = 'Col Charleston'
    starting_bracket[rnd][loc][14] = 'Bucknell'
    starting_bracket[rnd][loc][15] = 'Iona'
    starting_bracket[rnd][loc][16] = 'Penn'

    loc = 'east'
    starting_bracket[rnd][loc][1] = 'Villanova'
    starting_bracket[rnd][loc][2] = 'Purdue'
    starting_bracket[rnd][loc][3] = 'Texas Tech'
    starting_bracket[rnd][loc][4] = 'Wichita St'
    starting_bracket[rnd][loc][5] = 'West Virginia'
    starting_bracket[rnd][loc][6] = 'Florida'
    starting_bracket[rnd][loc][7] = 'Arkansas'
    starting_bracket[rnd][loc][8] = 'Virginia Tech'
    starting_bracket[rnd][loc][9] = 'Alabama'
    starting_bracket[rnd][loc][10] = 'Butler'
    starting_bracket[rnd][loc][11] = 'St Bonaventure'
    starting_bracket[rnd][loc][12] = 'Murray St'
    starting_bracket[rnd][loc][13] = 'Marshall'
    starting_bracket[rnd][loc][14] = 'SF Austin'
    starting_bracket[rnd][loc][15] = 'CS Fullerton'
    starting_bracket[rnd][loc][16] = 'Radford'
    return starting_bracket


def scoreBracket(realization, mlbracket):
    # takes in a given realization
    # and mlbracket which is your best bracket that you will be posting on ESPN.com (or other)

    score = 0
    for rnd in [2, 3, 4, 5]:
        rnd_ml = []
        rnd_real = []
        for region in realization[rnd].keys():
            for team in realization[rnd][region].values():
                rnd_real.append(team)
            for team in mlbracket[rnd][region].values():
                rnd_ml.append(team)
        num_correct = 0
        for team in rnd_ml:
            if team in rnd_real:
                num_correct += 1
        score += num_correct * round_scores[rnd - 1]

    num_correct = 0
    for team in mlbracket['Final']:
        if team in realization['Final']:
            num_correct += 1
    score += num_correct * round_scores[5]

    if mlbracket['Winner'] == realization['Winner']:
        score += round_scores[6]

    return score


def one_realization(predictions):
    # takes in all possible matchups and generates a random realization of the tournament

    bracket = get_starting_bracket()
    for rnd in [1, 2, 3, 4]:
        for region in bracket[1].keys():
            for s1, s2 in seed_pairs_by_round[rnd].items():
                rand = np.random.uniform(0, 1)
                team1, team2 = bracket[rnd][region][s1], bracket[rnd][region][s2]
                try:
                    # print(team1,team2)
                    prob = predictions[team1][team2]
                    flip = False
                except:
                    prob = predictions[team2][team1]
                    flip = True
                if flip:
                    prob = 1. - prob
                if rand < prob:
                    bracket[rnd + 1][region][s1] = team1
                else:
                    bracket[rnd + 1][region][s1] = team2

    #############################################################
    # Final Four
    rand = np.random.uniform(0, 1)
    team1, team2 = bracket[5]['south'][1], bracket[5]['west'][1]
    try:
        prob = predictions[team1][team2]
        flip = False
    except:
        prob = predictions[team2][team1]
        flip = True
    if flip:
        prob = 1. - prob
    if rand < prob:
        bracket['Final'].append(team1)
    else:
        bracket['Final'].append(team2)

    rand = np.random.uniform(0, 1)
    team1, team2 = bracket[5]['east'][1], bracket[5]['midwest'][1]
    try:
        prob = predictions[team1][team2]
        flip = False
    except:
        prob = predictions[team2][team1]
        flip = True
    if flip:
        prob = 1. - prob
    if rand < prob:
        bracket['Final'].append(team1)
    else:
        bracket['Final'].append(team2)

    #############################################################
    # Final
    rand = np.random.uniform(0, 1)
    team1, team2 = bracket['Final'][0], bracket['Final'][1]

    try:
        prob = predictions[team1][team2]
        flip = False
    except:
        prob = predictions[team2][team1]
        flip = True

    if flip:
        prob = 1. - prob
    if rand < prob:
        bracket['Winner'] = team1
    else:
        bracket['Winner'] = team2

    return bracket


# This is where we create our best possible bracket (mlbracket) to submit to ESPN.com (or other site)
# its just rounding our probabilities to take the team with highest probability as winner
rounded_submission_dict = {}
for team1i in team1:
    rounded_submission_dict[team1i.strip()] = {}
for team1i, team2i, prob in zip(team1, team2, probs):
    rounded_submission_dict[team1i.strip()][team2i.strip()] = round(prob)
mlbracket = one_realization(rounded_submission_dict)

winners = []
finalsteams = []
scores = []
gotwinnerright = []

numRealizations = 100000
for i in range(numRealizations):
    realization = one_realization(submission_dict)
    score = scoreBracket(realization, mlbracket)
    winners.append(realization['Winner'])
    finalsteams.append(realization['Final'])
    scores.append(score)
    if realization['Winner'] == mlbracket['Winner']:
        gotwinnerright.append(1)
    else:
        gotwinnerright.append(0)

scores = np.array(scores)
gotwinnerright = np.array(gotwinnerright, dtype='int')
correct_champ_scores = scores[gotwinnerright == 1]
wrong_champ_scores = scores[gotwinnerright == 0]

winners = np.array(winners, dtype='str')
winteams = []
wincounts = []
for t in np.unique(winners):
    winteams.append(t)
    wincounts.append(len(winners[winners == t]))
winteams = np.array(winteams, dtype='str')
wincounts = np.array(wincounts, dtype='float')

argsort = np.argsort(wincounts)

x = np.arange(len(winteams))

plt.figure(figsize=(12, 7))
plt.title('Tournament Winning Teams for each Realization', fontsize=25)
plt.ylabel('Fraction of Simulated Tournaments Won', fontsize=20)
plt.bar(x[-15:], height=wincounts[argsort][-15:] / numRealizations)
plt.xticks(x[-15:], winteams[argsort][-15:], rotation=90)  # no need to add .5 anymore
plt.gcf().subplots_adjust(bottom=0.25)
plt.show()

plt.clf()
plt.figure(figsize=(12, 7))
# We split up by correct champion or not because that is worth a lot of points
# and makes the distribution strongly bimodal
plt.title('Predicted Champion (%s) in %d Percent of Realizations' % (mlbracket['Winner'],
                                                                     (100 * float(wincounts[winteams == mlbracket[
                                                                         'Winner']]) / numRealizations)), fontsize=25)
plt.hist(wrong_champ_scores, bins=np.linspace(min(scores), max(scores), 200),
         label='Wrong Champion \nMean: %d \nStd: %d' % (np.mean(wrong_champ_scores),
                                                        np.std(wrong_champ_scores)), alpha=.8)
plt.hist(correct_champ_scores, bins=np.linspace(min(scores), max(scores), 200),
         label='Correct Champion! \nMean: %d \nStd: %d' % (np.mean(correct_champ_scores),
                                                           np.std(correct_champ_scores)), alpha=.8)
plt.xlabel('ESPN.com Score', fontsize=20)
plt.ylabel('# Realizations', fontsize=20)
plt.legend()
plt.show()
