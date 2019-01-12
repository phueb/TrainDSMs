import yaml
import shutil

from src import config


KEY = 'num_epochs'
VALUE = 10


ps = config.Dirs.runs.rglob('params.yaml')

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
        with p.open('r') as f:
            param2val = yaml.load(f)
            try:
                val = param2val[KEY]
            except KeyError:
                continue
            else:
                print(val)
                if val == VALUE:
                    location = p.parent
                    locations_to_del.append(location)

# delete
for loc in locations_to_del:
    print('Removing {}'.format(loc))
    shutil.rmtree(str(loc))