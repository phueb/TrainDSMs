async def fetch(session, url):
    async with session.get(url) as response:
        assert response.status == 200
        return await response.text()


def scrape_html(page):
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
    if len(synonyms_) > GlobalConfigs.NUM_SYNS:
        synonyms_ = synonyms_[:GlobalConfigs.NUM_SYNS]
    return synonyms_


async def get_syns(token_):
    url = 'http://www.thesaurus.com/browse/' + token_
    with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
    syns = scrape_html(html)
    return syns


# filter
tags = GlobalConfigs.POS_TAGS_DICT[pos]
filtered_token_list = [term for term, tags_d in self.hub.train_terms.term_tags_dict.items()
                       if term not in GlobalConfigs.SPECIAL_SYMBOLS + GlobalConfigs.SINGLE_LETTER_TERMS
                       and sorted(tags_d.items(), key=lambda i: i[1])[-1][0] in tags]
# make token_syns_dict
print('Scraping thesaurus.com...')
start = time.time()
loop = asyncio.get_event_loop()
synonyms_list = loop.run_until_complete(
    asyncio.gather(*[get_syns(t) for t in filtered_token_list]))
token_syns_dict = dict(zip(filtered_token_list, synonyms_list))
print('Made token_syns_dict in {:4.2f}'.format(time.time() - start))