import os
import configparser

config = configparser.ConfigParser()
config.read('./const.ini')
print(config.sections())

stage = config['const']['stage']
gender = config['const']['gender']

INDIR = f"../data/input/{stage}DataFiles{gender}"
SS = os.path.join(INDIR, f"SampleSubmission{stage}.csv")
print(SS)
OUTDIR = f"../data/output/{stage}{gender}"
SBMTDIR = "../data/output/sbmt"
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
