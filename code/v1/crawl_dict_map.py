# -*- coding: utf-8 -*-
'''
    Code to create a Dictionary Mapping from Cantonese to Standard Chinese
    Link to the Database: https://apps.itsc.cuhk.edu.hk/hanyu/Page/Cover.aspx
'''

import os
import configuration as config
import requests
from scrapy import Selector
import time
import random
import pickle
import re
from collections import defaultdict


MAJOR_URL = "https://apps.itsc.cuhk.edu.hk/hanyu/Page/"
HTML_DIR = os.path.join(config.DATA_PATH, "static", "chi_cuhk_database_2001")

if not os.path.exists(HTML_DIR):
    os.mkdir(HTML_DIR)

TERM_LIST = [
    "天象、地理",  # 164
    "時間、節令",  # 247
    "礦物、自然物",  # 27
    "動物",  # 353
    "植物、糧菜、果品",  # 400
    "飲食",  # 668
    "服飾",  # 322
    "房屋",  # 259
    "家具、日常用品",  # 457
    "工具、材料",  # 407
    "商業、交通",  # 618
    "文化、娛樂",  # 475
    "人體",  # 431
    "稱謂",  # 1324
    "動詞",  # 5246
    "形容詞",  # 2730
    "代詞",  # 164
    "數量詞",  # 215
    "副詞",  # 603
    "介詞",  # 64
    "連詞",  # 50
    "其他",  # 2364
]


def get_term_dict(verbose=False):
    '''
        output (dict):
            {term: {word: search-url-suffix}}
    '''
    term_dict = {}

    for term in TERM_LIST:
        path = os.path.join(HTML_DIR, "%s.html" % term)
        sel = Selector(text=open(path).read())

        word2url = {}
        for node in sel.xpath("//*[contains(@id, 'MainContent_repeaterTermsSearchResult_hlRecordID')]"):
            word, url = node.xpath('./span/text()').extract()[0], node.attrib['href']
            word2url[word] = url

        term_dict[term] = word2url
        if verbose: print("Count(%s): %d" % (term, len(word2url)))
    return term_dict


def get_interwords_dict(term_dict, word_list, verbose=True):
    ''' get the intersection words between word_list VS term_dict
        output (dict):
            {term: {intersect-word: url}
    '''
    print("Checking Intersection...")
    inter_dict = {}

    for term in TERM_LIST:
        if term not in term_dict:
            continue
        word2url = term_dict[term]
        inter = set(word2url.keys()).intersection(word_list)
        if verbose:
                print("%s: %d / %d (%.4f)" % (term, len(inter), len(word2url), len(inter)/len(word2url)))
        inter_dict[term] = {w: word2url[w] for w in list(inter)}
    if verbose:
        print("Total: %s/%s" % (
            sum([len(inter) for inter in inter_dict.values()]),
            sum([len(word2url) for word2url in term_dict.values()]))
        )
    return inter_dict


def crawl_word_html_from_term_dict(term_dict):
    for term in TERM_LIST:
        if term not in term_dict:
            continue
        word2url = term_dict[term]

        already_exists = os.listdir(os.path.join(HTML_DIR, term))
        need_crawl_files = [(w, search_suffix) for w, search_suffix in word2url.items()
                            if '%s.html' % search_suffix not in already_exists]
        print("Crawling words for %s (%d/%d) ..." % (term, len(need_crawl_files), len(word2url)))
        print("Estimated Time: %.2f-min" % (len(need_crawl_files) *2 / 60))
        for ind, (w, search_suffix) in enumerate(need_crawl_files):
            r = requests.get(MAJOR_URL + search_suffix)
            with open(os.path.join(HTML_DIR, term, "%s.html" % search_suffix), "wb") as f:
                f.write(r.content)
                print("(%s %d/%d) Crawled %s (%s)." % (term, ind, len(need_crawl_files), search_suffix, w))
            time.sleep(random.uniform(0.1, 0.5))


def parse_word_html(html_path):
    ''' output(dict):
            {canto-word(str): chinese explanation(list)}
    '''
    sel = Selector(text=open(html_path).read())
    canto_words = [res.extract() for res in sel.xpath(
        "//*[contains(@id, 'MainContent_repeaterRecord_lbl粵語詞彙')]/text()")]
    assert len(canto_words) == 1
    stdch_words = [res.extract() for res in sel.xpath(
                   "//*[contains(@id, 'MainContent_repeaterRecord_repeaterTranslation')]/text()")]
    return {canto_words[0]: stdch_words}


def save_canto_dict_map(canto_emb_file="canto_wiki.pkl",
                        output_file="canto2stdch.dict", apply_cleansing=True):
    ''' canto mapping with words in canto-embedding
        - subject to term-catelog defined above
        - canto-word with multiple meanings would be kept in a same list

        apply_cleansing:
            - remove meaningless
            - remove meanings that takes too many characters (>4)
            - remove meaninngs that is too long compared with the original word
    '''

    CLEANSING_PREFIX = [
        '(不是粵方言詞)', '(沒有對應詞彙)', '表示', '指',
    ]

    term_dict = get_term_dict()

    if canto_emb_file is not None:
        canto_embedding = pickle.load(
            open(os.path.join(config.EMB_PATH, canto_emb_file), "rb"))
        inter_dict = get_interwords_dict(term_dict, list(canto_embedding.keys()))
    else:
        inter_dict = term_dict  # use all terms

    canto_dict_map = defaultdict(list)
    # repeated_words = set()
    for term, word2url in inter_dict.items():
        for w, url in word2url.items():
            word_html = os.path.join(HTML_DIR, term, "%s.html" % url)
            if not os.path.exists(word_html):
                print("(%s, %s/%s) not exists!" % (term, w, url))
                continue

            canto_w, stdch_list = [(k, v) for k, v in parse_word_html(word_html).items()][0]
            # if canto_w in canto_dict_map:
            #     repeated_words.add(canto_w)
            #     continue
            clean_list = []
            for stdch in stdch_list:
                if apply_cleansing:
                    if any([stdch.startswith(x) for x in CLEANSING_PREFIX]):
                        continue
                    if len(stdch) > 4 or (len(stdch) - len(canto_w) > 2):
                        continue
                    stdch = re.sub(r'\(.*\)', '', stdch)
                    stdch = re.sub('\(|\)', '', stdch)
                clean_list.append(stdch)
            if clean_list:
                canto_dict_map[canto_w] += clean_list

    # remove words have multiple meanings
    # canto_dict_map = {k: v for k, v in canto_dict_map.items() if k not in repeated_words}

    with open(os.path.join(HTML_DIR, output_file), "w") as f:
        for k, v in sorted(canto_dict_map.items(), key=lambda x: (len(x[0]), len(x[1]))):
            f.write("%s %s\n" % (k, '/'.join(v)))
        print("Total%s: %d." % ('(apply cleansing)' if apply_cleansing else '', len(canto_dict_map)))
        print("%s saved." % f.name)


def filter_canto_dict_map(stdch_emb_file="stdch_wiki.pkl",
                          input_file="canto2stdch.dict",
                          simple=True):
    ''' filtering on dict-map file
        - if simple:
            - force 1-1 mapping (instead of multi-lines)
        - otherwise, also
            - remove unncessary canto-words (input)
            - select only stdch-words (output) valid in stdch-embeddings
    '''

    if not simple:
        stdch_embedding = pickle.load(
            open(os.path.join(config.EMB_PATH, stdch_emb_file), "rb"))
        stdch_set = set(stdch_embedding.keys())

    with open(os.path.join(HTML_DIR, input_file)) as input_f:
        with open(os.path.join(HTML_DIR, input_file + '.clean'), "w") as output_f:
            cnt = 0
            for l in input_f.readlines():
                canto_w, stdch_list = l[:-1].split(' ')
                if simple:
                    for stdch_w in sorted(stdch_list.split('/'), key=lambda x: len(x)):
                        output_f.write('%s %s\n' % (canto_w, stdch_w))
                        cnt += 1
                        break
                else:
                    if canto_w in stdch_set:
                        continue
                    for stdch_w in sorted(stdch_list.split('/'), key=lambda x: len(x)):
                        if stdch_w in stdch_set:
                            output_f.write('%s %s\n' % (canto_w, stdch_w))
                            cnt += 1
                            break
            print("Total(after filter): %d." % cnt)
            print("%s saved." % (output_f.name))

if __name__ == '__main__':

    print("\n\n\n === Create Terms Overview... ")
    term_url = MAJOR_URL + "Terms.aspx"
    r = requests.get(term_url)
    with open(os.path.join(HTML_DIR, "terms_overview.html"), "wb") as f:
        f.write(r.content)
        print("Crawled %s." % f.name)

    print("\n\n\n === Create Terms Dictionary... ")
    path = os.path.join(HTML_DIR, "terms_overview.html")
    sel = Selector(text=open(path).read())

    for res in sel.xpath("//*[contains(@href, 'Terms.aspx?target=')]/text()"):
        term = res.extract()

        try:
            r = requests.get(term_url + "?target=%s" % term)
        except Exception as e:
            print(e)
            continue

        term_dir = os.path.join(HTML_DIR, term)
        if not os.path.exists(term_dir):
            os.mkdir(term_dir)

            with open(os.path.join(HTML_DIR, "%s.html" % term), "wb") as f:
                f.write(r.content)
                print("Crawled %s." % f.name)
                time.sleep(random.randint(1, 3))

    #
    print("\n\n\n === Crawl pages (Intersection Terms Only)... ")

    canto_emb_file = "canto_wiki.pkl"
    canto_embedding = pickle.load(
        open(os.path.join(config.EMB_PATH, canto_emb_file), "rb"))

    term_dict = get_term_dict()
    # inter_dict = get_interwords_dict(term_dict, list(canto_embedding.keys()))
    inter_dict = term_dict
    crawl_word_html_from_term_dict(inter_dict)

    print("\n\n\n === Generate Canto-Stdch Dictionary Mapping...")
    save_canto_dict_map()
    filter_canto_dict_map()
