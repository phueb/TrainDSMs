import shutil

from src import config


ps = config.Dirs.runs.glob('*')

locations_to_del = []
while True:
    try:
        location = next(ps)
    except OSError as e:  # host is down
        raise OSError('Cannot access remote runs_dir. Check VPN and/or mount drive.')
    except StopIteration:
        print('Done')
        break
    else:
        print(location)
        try:
            f = (location / 'params.yaml').open('r')
        except FileNotFoundError:
            print('Does not have params.yaml')
            locations_to_del.append(location)

# delete
for loc in locations_to_del:
    print('Removing {}'.format(loc))
    shutil.rmtree(str(loc))