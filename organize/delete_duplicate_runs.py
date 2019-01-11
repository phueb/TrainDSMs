import yaml
import shutil

from src import config


ps = config.Dirs.runs.rglob('params.yaml')

param2vals = []
locations_to_del = []
while True:
    try:
        p = next(ps)
    except OSError as e:  # host is down
        raise OSError('Cannot access remote runs_dir. Check VPN and/or mount drive.')
    except StopIteration:
        print('Done')
        break
    else:
        print(p)
        with p.open('r') as f:
            param2val = yaml.load(f)
        if param2val not in param2vals:
            param2vals.append(param2val)
        else:
            print('Is a duplicate')
            location = p.parent
            locations_to_del.append(location)
# delete
for loc in locations_to_del:
    print('Removing {}'.format(loc))
    shutil.rmtree(str(loc))