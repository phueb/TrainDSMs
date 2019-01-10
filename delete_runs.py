import yaml
from itertools import chain
import shutil

from src import config


KEY = 'count_type'
VALUE = 'ww'


ps = chain(config.Dirs.runs.rglob('params.yaml'),
           config.Ludwig.runs_dir.rglob('params.yaml'))

locations_to_del = []
while True:
    try:
        p = next(ps)
    except OSError as e:  # host is down
        if config.Ludwig.exit_on_error:
            print(e)
            raise OSError('Cannot access remote runs_dir. Check VPN and/or mount drive.')
        else:
            print(e)
            print('WARNING: Cannot access remote runs_dir. Check VPN and/or mount drive.')
            pass
    except StopIteration:
        print('Done')
        break
    else:
        with p.open('r') as f:
            param2val = yaml.load(f)
            try:
                val = param2val[KEY]
            except KeyError:
                continue
            else:
                print(val[0])
                if val[0] == VALUE:
                    location = p.parent
                    locations_to_del.append(location)

# delete
for loc in locations_to_del:
    print('Removing {}'.format(loc))
    shutil.rmtree(str(loc))