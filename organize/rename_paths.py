
from src import config

SRC_NAME = 'parm2val.yaml'
DST_NAME = 'param2val.yaml'


for p in config.Dirs.backup.rglob(SRC_NAME):
    print(p)
    p.rename(p.parent / DST_NAME)

