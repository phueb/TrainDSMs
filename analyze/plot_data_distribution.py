import matplotlib.pyplot as plt
import numpy as np

from two_stage_nlp import config

DATA_NAME1 = 'cohyponyms'
DATA_NAME2 = 'syntactic'

def load_probes():
    data_dir = '{}/{}'.format(DATA_NAME1, DATA_NAME2) if DATA_NAME2 is not None else DATA_NAME1
    p = config.Dirs.tasks / data_dir / '{}_{}.txt'.format(
        config.Corpus.name, config.Corpus.num_vocab)
    probes = []
    probe_relata = []
    with p.open('r') as f:
        for line in f.read().splitlines():
            spl = line.split()
            probe = spl[0]
            relata = spl[1:]
            probes.append(probe)
            probe_relata.append(relata)
    return probes, probe_relata


# data
probes, probe_relata = load_probes()
num_total_probes = len(probes)
num_unique_probes = len(np.unique(probes))
num_unique_relata = len(np.unique(np.concatenate(probe_relata)))

print(DATA_NAME1, DATA_NAME2)
print('Num total probes={}'.format(num_total_probes))
print('Num unique probes={}'.format(num_unique_probes))
print('Num unique relata={}'.format(num_unique_relata))
print('Min num_relata={}'.format(min([len(relata) for relata in probe_relata])))
print('Max num_relata={}'.format(max([len(relata) for relata in probe_relata])))


# fig
fig, ax = plt.subplots(1)
plt.title('Distribution of {} {} relata'.format(DATA_NAME1, DATA_NAME2))
pr, cs = np.unique(np.concatenate(probe_relata), return_counts=True)
ax.plot(np.sort(cs))
ax.set_xlabel('Relata sorted by Frequency')
ax.set_ylabel('Frequency')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([])
ax.set_xticklabels([])
fig.tight_layout()
plt.show()