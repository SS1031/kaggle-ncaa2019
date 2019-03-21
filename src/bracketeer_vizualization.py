import os
from bracketeer import build_bracket
import CONST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sbmt_path', default='')
options = parser.parse_args()

sbmt_path = options.sbmt_path
out_name = os.path.basename(sbmt_path).split('.')[0] + '.png'

b = build_bracket(
    outputPath=os.path.join(CONST.OUTDIR, out_name),
    teamsPath=os.path.join(CONST.INDIR, 'Teams.csv'),
    seedsPath=os.path.join(CONST.INDIR, 'NCAATourneySeeds.csv'),
    submissionPath=sbmt_path,
    slotsPath=os.path.join(CONST.INDIR, 'NCAATourneySlots.csv'),
    year=2019
)
