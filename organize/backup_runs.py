

from src import config
from src.jobs import backup_job

for location in config.Dirs.runs.glob('**/*num*'):
    param_name, job_name = location.parts[-2:]
    print(param_name, job_name)
    backup_job(param_name, job_name, allow_rewrite=True)
    print()