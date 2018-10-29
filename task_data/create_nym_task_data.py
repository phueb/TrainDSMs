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
NYM_TYPE = 'antonym'
POS = 'adj'  # noun, verb, adj (not adjective)
LEMMATIZE = True
NUM_NYMS = 5
VERBOSE = True


EXCLUDED = {'verb': ['do', 'is', 'be', 'wow', 'was', 'did', 'are',
                     'let', 'am', 'cow', 'got', 'woo', 'squirrel',
                     'lets', 'wolf', 'harry', 'market', 'tires', 'crane',
                     'neigh', 'parrot', 'waffle', 'flounder', 'fries',
                     'squirrels', 'clunk', 'microwave', 'dong', 'paw',
                     'potter', 'spout', 'telescope', 'bumps', 'vest',
                     'pine', 'sack', 'ax', 'cluck', 'fudge', 'ships',
                     'josh', 'duck', 'spoon', 'boo', 'diaper', 'shoulder',
                     'sock', 'jimmy'],
            'noun': ['it', 'she', 'he', 'pas', 'tom', 'pooh', 'doing',
                     'yeah', 'mine', 'find', 'win', 'ruff', 'er', 'ah',
                     'go', 'mis', 'lee', 'jay', 'smith', 'leaning', 'might',
                     'rex', 'fix', 'ugh', 'fred', 'pussy', 'mot', 'um', 'oop',
                     'sh', 'pail', 'mr', 'will', 'fill', 'snapping', 'meg',
                     'victor', 'joe', 'foo', 'wait', 'phooey', 'ninny', 'sonny',
                     'valentine', 'po', 'moira'],
            'adj': []}


async def get_nyms(w):
    with aiohttp.ClientSession() as session:
        html = await fetch(session, w)
    nyms = scrape_nyms(html, w)
    return nyms


async def fetch(session, w, verbose=VERBOSE):
    url = 'http://www.thesaurus.com/browse/' + w
    if verbose:
        print('Fetching from {}'.format(url))
    async with session.get(url) as response:
        if response.status != 200 and verbose:
            print('WARNING: Did not reach {}'.format(url))
        return await response.text()


def scrape_nyms(page, w, verbose=VERBOSE):  # TODO collapse all POS for antonyms?
    if verbose:
        print('\nScraping {}s for "{}"'.format(NYM_TYPE, w))
    soup = BeautifulSoup(page, 'lxml')
    try:
        section = soup.find_all('section', {'class': 'synonyms-container'})[0]  # TODO iterate over multiple sections
    except IndexError:
        return []

    try:
        found_pos = section.select('em')[0].text.strip()  # need to strip
        if verbose:
            print('\tFound "{}" section'.format(found_pos))
    except IndexError:
        if verbose:
            print('\tSkipping. No POS info available.')  # TODO better way to find verb section when noun section is found first
        return []
    else:
        if POS == found_pos:
            if NYM_TYPE == 'synonym':
                res = find_synonyms(section)
            elif NYM_TYPE == 'antonym':
                try:
                    section = soup.find_all('section', {'class': 'antonyms-container'})[0]
                except IndexError:  # no antonyms
                    print('No antonyms found')
                    res = []
                else:
                    res = find_antonyms(section)
            else:
                raise AttributeError('Invalid arg to "nym_type".')
            #
            if len(res) > NUM_NYMS:
                res = res[:NUM_NYMS]
            return res
        else:
            return []


def find_synonyms(section):
    res = []
    for li in section.find_all('li'):
        if len(li.text.split()) == 1:
            res.append(li.text)
    return res


def find_antonyms(section):
    res = []
    for a in section.select('a'):
        antonym = a.text.strip()
        res.append(antonym)
    return res


if __name__ == '__main__':
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    for vocab_size in config.Task.vocab_sizes:
        vocab = EmbedderBase.load_corpus_data(num_vocab=vocab_size)[1]
        assert len(vocab) == vocab_size
        probes = []
        for w in vocab:
            if len(w) > 1:
                if w[0] not in list(string.punctuation) \
                        and w[1] not in list(string.punctuation) \
                        and w not in EXCLUDED[POS]:
                    if LEMMATIZE:
                        w = lemmatizer(w, POS)[0]
                    probes.append(w)
        if LEMMATIZE:
            probes = set([p for p in probes if p in vocab])  # lemmas may not be in vocab
        out_path = config.Dirs.tasks / '{}_{}s'.format(POS, NYM_TYPE) / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
        if not out_path.parent.exists():
            out_path.parent.mkdir()
        with out_path.open('w') as f:
            for probes_partition in itertoolz.partition(100, probes):  # web scraping must be done in chunks
                loop = asyncio.get_event_loop()
                nyms_list = loop.run_until_complete(asyncio.gather(*[get_nyms(w) for w in probes_partition]))
                # write to file
                print('Writing {}'.format(out_path))
                for probe, nyms in zip(probes_partition, nyms_list):
                    for nym in nyms:
                        if nym in vocab and nym != probe:
                            line = '{} {}\n'.format(probe, nym)
                            print(line.strip('\n'))
                            f.write(line)