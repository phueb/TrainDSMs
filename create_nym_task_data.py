from bs4 import BeautifulSoup
import aiohttp
import string
import asyncio
from cytoolz import itertoolz
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

from src import config
from src.embedders.base import EmbedderBase

CORPUS_NAME = 'childes-20180319'
LEMMATIZE = True
VERBOSE = False


EXCLUDED = ['do', 'is', 'be', 'wow', 'was', 'did', 'are',
            'let', 'am', 'cow', 'got', 'woo', 'squirrel',
            'lets', 'wolf', 'harry', 'market', 'tires', 'crane',
            'neigh', 'parrot', 'waffle', 'flounder', 'fries',
            'squirrels', 'clunk', 'microwave', 'dong', 'paw',
            'potter', 'spout', 'telescope', 'bumps', 'vest',
            'pine', 'sack', 'ax', 'cluck', 'fudge', 'ships',
            'josh', 'duck', 'spoon', 'boo', 'diaper', 'shoulder',
            'sock', 'jimmy', 'it', 'she', 'he', 'pas', 'tom', 'pooh', 'doing',
            'yeah', 'mine', 'find', 'win', 'ruff', 'er', 'ah',
            'go', 'mis', 'lee', 'jay', 'smith', 'leaning', 'might',
            'rex', 'fix', 'ugh', 'fred', 'pussy', 'mot', 'um', 'oop',
            'sh', 'pail', 'mr', 'will', 'fill', 'snapping', 'meg',
            'victor', 'joe', 'foo', 'wait', 'phooey', 'ninny', 'sonny',
            'valentine', 'po', 'moira']


async def get_nyms(w, nym_type):
    with aiohttp.ClientSession() as session:
        html = await fetch(session, w)
    nyms = scrape_nyms(html, w, nym_type)
    return nyms


async def fetch(session, w, verbose=VERBOSE):
    url = 'http://www.thesaurus.com/browse/' + w
    if verbose:
        print('Fetching from {}'.format(url))
    async with session.get(url) as response:
        if response.status != 200 and verbose:
            print('WARNING: Did not reach {}'.format(url))
        return await response.text()


def scrape_nyms(page, w, nym_type, verbose=VERBOSE):
    if verbose:
        print('\nScraping for "{}"'.format(w))
    soup = BeautifulSoup(page, 'lxml')
    #
    if nym_type == 'syn':
        res = find_synonyms(soup)
    elif nym_type == 'ant':
        res = find_antonyms(soup)
    else:
        raise AttributeError('Invalid arg to "nym_type".')
    return res


def find_synonyms(soup):
    res = []
    try:
        section = soup.find_all('section', {'class': 'synonyms-container'})[0]  # TODO iterate over multiple sections?
    except IndexError:
        print('No synonyms found')
        return []
    else:
        for li in section.find_all('li'):
            if len(li.text.split()) == 1:
                synonym = li.text.strip()
                res.append(synonym)
    return res


def find_antonyms(soup):
    res = []
    try:
        section = soup.find_all('section', {'class': 'antonyms-container'})[0]
    except IndexError:
        print('No antonyms found')
        return []
    else:
        for a in section.select('a'):
            antonym = a.text.strip()
            res.append(antonym)
    return res


if __name__ == '__main__':
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    for vocab_size in config.Eval.vocab_sizes:
        vocab = EmbedderBase.load_corpus_data(num_vocab=vocab_size)[1]
        assert len(vocab) == vocab_size
        probes = []
        for w in vocab:
            if len(w) > 1:
                if w[0] not in list(string.punctuation) \
                        and w[1] not in list(string.punctuation) \
                        and w not in EXCLUDED:
                    if LEMMATIZE:
                        for pos in ['noun', 'verb', 'adj']:
                            w = lemmatizer(w, pos)[0]
                            probes.append(w)
                    else:
                        probes.append(w)
        if LEMMATIZE:
            probes = set([p for p in probes if p in vocab])  # lemmas may not be in vocab
        #
        for nym_type in ['syn', 'ant']:
            out_path = config.Dirs.tasks / 'nyms' / nym_type / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            if not out_path.parent.exists():
                out_path.parent.mkdir()
            with out_path.open('w') as f:
                for probes_partition in itertoolz.partition(100, probes):  # web scraping must be done in chunks
                    loop = asyncio.get_event_loop()
                    nyms = loop.run_until_complete(asyncio.gather(*[get_nyms(w, nym_type) for w in probes_partition]))
                    # write to file
                    print('Writing {}'.format(out_path))
                    for probe, nyms in zip(probes_partition, nyms):
                        nyms = ' '.join([nym for nym in nyms
                                         if nym != probe and nym in vocab])
                        if not nyms:
                            continue
                        line = '{} {}\n'.format(probe, nyms)
                        print(line.strip('\n'))
                        f.write(line)