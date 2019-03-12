
from two_process_nlp.jobs import preprocessing_job

for vocab_size in [16384]:
    preprocessing_job(vocab_size, skip_docs=True)