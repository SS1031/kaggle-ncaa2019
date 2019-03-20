import os
from bracketeer import build_bracket
import CONST


sub_path = 'config0016.csv'
out_name = sub_path.split('.')[0] + '.png'
b = build_bracket(
    outputPath=os.path.join(CONST.OUTDIR, 'bracketeer', out_name),
    teamsPath=os.path.join(CONST.INDIR, 'Teams.csv'),
    seedsPath=os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'),
    submissionPath=os.path.join(CONST.SBMTDIR, 'config008.csv'),
    slotsPath=os.path.join(CONST.INDIR, 'NCAATourneySlots.csv'),
    year=2018
)
