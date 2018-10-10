from bs4 import BeautifulSoup
import aiohttp
import string
import asyncio
import time

from src import config


async def fetch(session, url):
    async with session.get(url) as response:
        assert response.status == 200
        return await response.text()


def scrape_html(page, pos):  # TODO how to get anynyms?
    # filters
    soup = BeautifulSoup(page, 'lxml')
    divs = soup.select("div.filters")
    # collect
    synonyms_ = []
    for div in divs:
        try:
            pos_ = div.select("em.txt")[0].getText()
        except IndexError:  # if verb is not recognized by thesaurus.com
            synonyms_ += ['NOT_RECOGNIZED']
        else:
            if pos_ == pos:
                synonyms_ += [li.select('span.text')[0].getText() for li in div.findAll("li")]
            else:
                synonyms_ += ['POS_MISMATCH']
    # num_syns
    if len(synonyms_) > config.NymMatching.num_nyms:
        synonyms_ = synonyms_[:config.NymMatching.num_nyms]
    return synonyms_


async def get_syns(w, pos):
    url = 'http://www.thesaurus.com/browse/' + w
    with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
    nyms = scrape_html(html, pos)
    return nyms

for pos in ['verb']:
    # filter vocab by POS
    tags = GlobalConfigs.POS_TAGS_DICT[pos]
    filtered_vocab = [w for w in self.hub.train_terms.term_tags_dict.items()
                      if w not in list(string.ascii_letters)]
    # make token_syns_dict
    print('Scraping thesaurus.com...')
    start = time.time()
    loop = asyncio.get_event_loop()
    synonyms_list = loop.run_until_complete(
        asyncio.gather(*[get_syns(t, pos) for t in filtered_vocab]))
    token_syns_dict = dict(zip(filtered_vocab, synonyms_list))
    for k, v in token_syns_dict.items():
        print(k, v)
    print('Made token_syns_dict in {:4.2f}'.format(time.time() - start))