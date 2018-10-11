from bs4 import BeautifulSoup
import aiohttp
import numpy as np
import asyncio
import time

from src import config
from src.utils import make_w2freq

CORPUS_NAME = 'childes-20180319'


async def fetch(session, w):
    url = 'http://www.thesaurus.com/browse/' + w
    print('Fetching from {}'.format(url))
    async with session.get(url) as response:
        assert response.status == 200
        return await response.text()


def scrape_nyms(page, pos, w):  # TODO how to get antonyms?
    print('\nScraping nyms for "{}"'.format(w))
    res = []
    soup = BeautifulSoup(page, 'lxml')
    for section in soup.find_all('section', {'class': 'synonyms-container'}):
        try:
            found_pos = section.select('em')[0].text.strip()  # need to strip
            print('\tFound"{}" section'.format(found_pos))
        except IndexError:
            pass
            print('\tSkipping section. No POS info available.')  # TODO this happens in noun sections only

            # TODO there must b better way to find verb section when noun section is found first

        else:
            if pos == found_pos:
                for li in section.find_all('li'):
                    if len(li.text.split()) == 1:
                        res.append(li.text)
                        # print(li.text)
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
        vocab = [w for w, f in w2freq.most_common(vocab_size - 1)]
        for pos, nym_type in [('verb', 'synonym')]:
            task_name = '{}_{}_matching'.format(pos, nym_type)
            out_fname = '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            out_path = config.Global.task_dir / task_name / out_fname
            with out_path.open('w') as f:
                for i in [1, 2]:  # web scraping must be done in chunks, otherwise connection reset
                    in_path = config.Global.task_dir / task_name / '{}.txt'.format(i)
                    probes = np.loadtxt(in_path, dtype='str')
                    # make w2nyms
                    print('Scraping thesaurus.com...')
                    start = time.time()
                    loop = asyncio.get_event_loop()
                    nyms = loop.run_until_complete(asyncio.gather(*[get_nyms(p, pos) for p in probes]))
                    # write to file
                    print('Writing {}'.format(out_path))
                    for probe, nyms in zip(probes, nyms):
                        if probe in vocab and nyms:
                            line = '{} {}\n'.format(probe, ' '.join(nyms))
                            f.write(line)