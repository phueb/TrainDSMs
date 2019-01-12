
from src.jobs import aggregation_job

EVAL_NAME = 'matching'


df = aggregation_job(EVAL_NAME)
print(df)