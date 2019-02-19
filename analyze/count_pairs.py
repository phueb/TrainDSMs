from itertools import product

from two_stage_nlp import config


CORPUS_NAME = config.Corpus.name
NUM_VOCAB = config.Corpus.num_vocab

num_row_words_sum = 0
num_total_possible_sum = 0
pos_prob_sum = 0
neg_prob_sum = 0
num_tasks = 0

for p in config.Dirs.tasks.rglob('{}_{}*.txt'.format(CORPUS_NAME, NUM_VOCAB)):

    with p.open('r') as f:
        lines = f.read().splitlines()  # removes '\n' newline character

    pairs = set()
    pairs.update()
    num_possible = 0
    probes = set()
    for line in lines:
        probe = line.split()[0]
        relata = line.split()[1:]
        pairs.update(list(product([probe], relata)))
        probes.update([probe] + relata)
        num_possible += len(relata)
    print(p.relative_to(config.Dirs.tasks))
    #
    num_unique = len(probes)
    print('Num unique probes={}'.format(num_unique))
    print('Average num relata per probe={}'.format(num_possible / len(lines)))
    #
    num_total_possible = len(probes) ** 2
    print('Num unique total possible={:,}'.format(num_total_possible))
    #
    num_pos = len(pairs)
    print('Num unique positive pairs={:,}'.format(num_pos))
    print('Num unique positive possible={:,}'.format(num_possible))
    print('Positive prob={:.3f}'.format(num_pos / num_total_possible))
    #
    num_neg = num_total_possible - num_pos
    print('Num unique negative possible={:,}'.format(num_neg))
    print('Negative prob={:.3f}'.format(num_neg / num_total_possible))
    #
    diff = num_possible - len(pairs)
    if diff > 0:
        print('WARNING: Difference={}. Duplicates pairs exist'.format(diff))
    print()
    # collect
    num_row_words_sum += len(lines)
    num_total_possible_sum += num_total_possible
    pos_prob_sum += num_pos / num_total_possible
    neg_prob_sum += num_neg / num_total_possible
    num_tasks += 1

print('Average num row_words per task={}'.format(num_row_words_sum / num_tasks))
print('Average num_total_possible per task={}'.format(num_total_possible_sum / num_tasks))
print('Average pos_prob per task={}'.format(pos_prob_sum / num_tasks))
print('Average neg_prob per task={}'.format(neg_prob_sum / num_tasks))
print('Average neg_pos_ratio={}'.format((neg_prob_sum / num_tasks) / (pos_prob_sum / num_tasks)))