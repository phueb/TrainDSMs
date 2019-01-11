
from src import config

SRC_NAME = 'events_'
DST_NAME = 'events'


for p in config.Dirs.runs.rglob(SRC_NAME):
    print(p)
    p.rename(p.parent / DST_NAME)

