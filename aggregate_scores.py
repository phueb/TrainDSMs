
from src.aggregator import Aggregator


ag_matching = Aggregator('matching')
matching_df = ag_matching.make_df()
print(matching_df)
matching_df.to_csv('{}.csv'.format(ag_matching.ev_name))