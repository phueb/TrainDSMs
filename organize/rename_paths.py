
from two_stage_nlp import config

SRC_NAME = 'parm2val.yaml'
DST_NAME = 'param2val.yaml'


for p in config.Dirs.remote_runs.rglob(SRC_NAME):
    print(p)
    p.rename(p.parent / DST_NAME)

