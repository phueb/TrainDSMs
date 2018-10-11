from bs4 import BeautifulSoup
import aiohttp
import string
import asyncio
from cytoolz import itertoolz

from src import config
from src.utils import make_w2freq

CORPUS_NAME = 'childes-20180319'
NYM_TYPE = 'synonym'  # TODO test antonym
POS = 'verb'

EXCLUDED = {'verb': ['do', 'is', 'be', 'wow', 'was', 'did', 'are',
                     'let', 'am', 'cow', 'got', 'woo', 'squirrel',
                     'lets', 'wolf', 'harry', 'market', 'tires', 'crane',
                     'neigh', 'parrot', 'waffle', 'flounder', 'fries',
                     'squirrels', 'clunk', 'microwave', 'dong', 'paw',
                     'potter', 'spout', 'telescope', 'bumps', 'vest',
                     'pine', 'sack', 'ax', 'cluck', 'fudge', 'ships',
                     'josh', 'duck', 'spoon', 'boo', 'diaper']}

async def fetch(session, w, verbose=False):
    url = 'http://www.thesaurus.com/browse/' + w
    if verbose:
        print('Fetching from {}'.format(url))
    async with session.get(url) as response:
        if response.status != 200:
            print('WARNING: Did not reach {}'.format(url))
        return await response.text()


def scrape_nyms(page, pos, w, verbose=False):
    if verbose:
        print('\nScraping nyms for "{}"'.format(w))
    res = []
    soup = BeautifulSoup(page, 'lxml')
    for section in soup.find_all('section', {'class': '{}s-container'.format(NYM_TYPE)}):
        try:
            found_pos = section.select('em')[0].text.strip()  # need to strip
            if verbose:
                print('\tFound"{}" section'.format(found_pos))
        except IndexError:
            if verbose:
                print('\tSkipping section. No POS info available.')

            # TODO this happens in noun sections only
            # TODO there must b better way to find verb section when noun section is found first

        else:
            if pos == found_pos:
                for li in section.find_all('li'):
                    if len(li.text.split()) == 1:
                        res.append(li.text)
                break

    # num_syns
    if len(res) > config.NymMatching.num_nyms:
        res = res[:config.NymMatching.num_nyms]
    return res


async def get_nyms(w, pos):
    with aiohttp.ClientSession() as session:
        html = await fetch(session, w)
    nyms = scrape_nyms(html, pos, w)
    return nyms


if __name__ == '__main__':
    w2freq = make_w2freq(CORPUS_NAME)
    for vocab_size in config.Tasks.vocab_sizes:
        vocab_ = [w for w, f in w2freq.most_common(vocab_size - 1)]
        vocab = []
        for w in vocab_:
            if len(w) > 1:
                if w[0] not in list(string.punctuation) \
                        and w[1] not in list(string.punctuation) \
                        and w not in EXCLUDED[POS]:
                    vocab.append(w)
        task_name = '{}_{}_matching'.format(POS, NYM_TYPE)
        out_fname = '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
        out_path = config.Global.task_dir / task_name / out_fname
        with out_path.open('w') as f:
            for vocab_partition in itertoolz.partition(100, vocab):  # web scraping must be done in chunks
                # make w2nyms
                loop = asyncio.get_event_loop()
                nyms_list = loop.run_until_complete(asyncio.gather(*[get_nyms(w, POS) for w in vocab_partition]))
                # write to file
                print('Writing {}'.format(out_path))
                for w, nyms in zip(vocab_partition, nyms_list):
                    if nyms:
                        line = '{} {}\n'.format(w, ' '.join(nyms))
                        print(line)
                        f.write(line)