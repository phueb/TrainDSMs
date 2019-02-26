import shutil

from two_process_nlp import config


param_ps = config.Dirs.remote_runs.glob('**/*num*')

locations_to_del = []
while True:
    try:
        location = next(param_ps)
    except OSError as e:  # host is down
        raise OSError('Cannot access remote runs_dir. Check VPN and/or mount drive.')
    except StopIteration:
        print('Done')
        break
    else:
        print(location)
        if not (location / 'embeddings.txt').exists():
            print('Does not have embeddings.')
            locations_to_del.append(location)

# delete
for loc in locations_to_del:
    print('Removing {}'.format(loc))
    # shutil.rmtree(str(loc))