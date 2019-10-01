
from two_process_nlp.job import preprocessing_job
from two_process_nlp import config

SKIP_DOCS = False
LOCAL = True

for vocab_size in config.Corpus.vocab_sizes:
    preprocessing_job(vocab_size, skip_docs=SKIP_DOCS, local=LOCAL)