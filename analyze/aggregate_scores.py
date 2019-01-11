
from src.aggregator import Aggregator

EVAL_NAME = 'matching'


ag_matching = Aggregator(EVAL_NAME)
matching_df = ag_matching.make_df()
print(matching_df)
matching_df.to_csv('{}.csv'.format(ag_matching.ev))