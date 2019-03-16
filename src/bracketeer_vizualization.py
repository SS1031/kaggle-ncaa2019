import os
from bracketeer import build_bracket
import CONST

b = build_bracket(
    outputPath=os.path.join(CONST.OUTDIR, 'bracketeer', 'output.png'),
    teamsPath=os.path.join(CONST.INDIR, 'Teams.csv'),
    seedsPath=os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'),
    submissionPath=os.path.join(CONST.SBMTDIR, 'config008.csv'),
    slotsPath=os.path.join(CONST.INDIR, 'NCAATourneySlots.csv'),
    year=2018
)
