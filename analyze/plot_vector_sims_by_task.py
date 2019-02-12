

from two_stage_nlp import config
from two_stage_nlp.job_utils import init_embedder
from two_stage_nlp.job_utils import w2e_to_sims
from two_stage_nlp.architectures import comparator
from two_stage_nlp.evaluators.matching import Matching
from two_stage_nlp.aggregator import Aggregator


for location in config.Dirs.remote_runs.glob('**/*num*'):
    param_name, job_name = location.parts[-2:]
    param2val = Aggregator.load_param2val(param_name)
    param2val['job_name'] = job_name
    #
    print(param2val['param_name'])
    embedder = init_embedder(param2val)
    embedder.load_w2e()

    # tasks
    for ev in [
                Matching(comparator, 'cohyponyms', 'semantic'),
                # Matching(comparator, 'cohyponyms', 'syntactic'),
                Matching(comparator, 'features', 'is'),
                Matching(comparator, 'features', 'has'),
                Matching(comparator, 'nyms', 'syn', suffix='_jw'),
                Matching(comparator, 'nyms', 'ant', suffix='_jw'),
                Matching(comparator, 'hypernyms'),
                Matching(comparator, 'events')
        ]:

        sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)

        print(sims_mat.shape)
        print(sims_mat.mean())


    raise SystemExit('just one for now')