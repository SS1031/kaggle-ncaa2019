import os
import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read('./const.ini')

stage = config['const']['stage']
gender = config['const']['gender']

INDIR = f"../data/input/{stage}DataFiles{gender}"
if stage == 'Stage2':
    SS = os.path.join(INDIR, "SampleSubmission.csv")
    if not os.path.exists(SS):
        s1 = pd.read_csv(os.path.join(INDIR, "SampleSubmissionStage1.csv"))
        s2 = pd.read_csv(os.path.join(INDIR, "SampleSubmissionStage2.csv"))
        pd.concat([s1, s2], axis=0).to_csv(SS, index=False)
else:
    SS = os.path.join(os.path.join(INDIR, "SampleSubmissionStage1.csv"))

OUTDIR = f"../data/output/{stage}{gender}"
SBMTDIR = f"../data/output/{stage}{gender}"
FEATDIR = f"../data/feature/{stage}{gender}"
SELECTEDDIR = os.path.join(FEATDIR, "selected")
TRNFEATDIR = os.path.join(FEATDIR, "trn")
TSTFEATDIR = os.path.join(FEATDIR, "tst")

EX_COLS = ['Season', 'T1TeamID', 'T2TeamID', 'Result']

if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)
if not os.path.exists(SBMTDIR): os.makedirs(OUTDIR)
if not os.path.exists(SELECTEDDIR): os.makedirs(SELECTEDDIR)
if not os.path.exists(TRNFEATDIR): os.makedirs(TRNFEATDIR)
if not os.path.exists(TSTFEATDIR): os.makedirs(TSTFEATDIR)
