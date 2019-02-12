
from two_stage_nlp import config
from two_stage_nlp.aggregator import Aggregator


def gen_param2vals_for_completed_jobs():
    for location in config.Dirs.remote_runs.glob('**/*num*'):
        param_name, job_name = location.parts[-2:]
        param2val = Aggregator.load_param2val(param_name)
        param2val['job_name'] = job_name
        yield param2val