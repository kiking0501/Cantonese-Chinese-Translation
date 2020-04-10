import os
from collections import defaultdict
from configuration import WIKI_PATH, WIKI_ORI_PATH
from determine_lang import DetermineLanguage
import json
import re
import jieba

if not os.path.exists(WIKI_ORI_PATH):
    raise IOError("The 'json' folder is missing!"
                  " You should"
                  " (1) first download a .xml.bz2 dump-file\n"
                  "     from https://dumps.wikimedia.org/zh_yuewiki/\n"
                  " (2) and then use the WikiExtractor\n"
                  "     https://github.com/attardi/wikiextractor\n"
                  "      to extract the wiki articles as json files\n"
                  " (3) and finally place the json files inside a 'json' folder\n"
                  "     under %s!" % WIKI_PATH)

DL = DetermineLanguage()


def load_jieba(dict_txt="dict.txt.pycanto-stdch_wiki", verbose=False):
    ''' load jieba with the stated dictionary txt (valid after ver 0.28 '''
    if dict_txt is None:
        dict_txt = "dict.txt.big"
    jieba.set_dictionary(os.path.join(DATA_PATH, "jieba_dict", dict_txt))
    if verbose:
        print("Successfully set Jieba-dict to '%s'. " % dict_txt)


def read_wiki(file_name):
    data = []
    with open(os.path.join(WIKI_ORI_PATH, file_name), 'r') as f:
        for json_obj in f:
            data.append(json.loads(json_obj))
    return data


def save_clean_wiki(file_name, verbose=True):
    json_list = read_wiki(file_name)
    for json_obj in json_list:
        output_path = os.path.join(WIKI_PATH, "clean", "%s_%s" % (json_obj['id'], json_obj['title'].replace('/', '-').strip()))
        with open(output_path, "w") as f:
            for line in json_obj['text'].split('\n'):
                content = re.findall('\（.*?\）', line)
                for c in content:
                    if DL.lang(c) in ['SYMBOLS', 'ENGLISH']:
                        line = line.replace(c, '')

                for l in line.split('。'):
                    if l:
                        f.write(' '.join(jieba.cut(l, cut_all=False)) + ' 。\n')
        print("%s saved." % output_path)


def read_clean_wiki(file_code):
    with open(os.path.join(WIKI_PATH, "clean", file_code)) as f:
        return [l.strip().split(' ') for l in f]


def save_clean_wikipedia(output_file="wiki_yue_overview.csv", verbose=True):
    load_jieba()
    for (_, _, filenames) in sorted(os.walk(WIKI_ORI_PATH)):
        for file_name in sorted(filenames):
            save_clean_wiki(file_name)
    total = 0
    with open(os.path.join(WIKI_PATH, output_file), "w") as f:
        file_codes = []
        for (_, _, filenames) in os.walk(os.path.join(WIKI_PATH, "clean")):
            file_codes.extend(filenames)
        for ind, code in enumerate(sorted(file_codes)):
            f.write('%s,%s\n' % (ind, code))
        total += len(file_codes)
    if verbose:
        print("Total: %d. %s saved." % (total, output_file))


def read_clean_wikipedia(overview_csv="wiki_yue_overview.csv"):
    with open(os.path.join(WIKI_PATH, overview_csv)) as f:
        for line in f.readlines():
            # print(line)
            _, file_code = line.partition(',')[0], line.partition(',')[2].strip()
            if file_code.startswith('wiki'):
                continue
            # print("Reading %s.." % file_code)
            for sen in read_clean_wiki(file_code):
                yield sen

if __name__ == '__main__':
    save_clean_wikipedia()

    tokdict = defaultdict(int)
    for sen in read_clean_wikipedia():
        for w in sen:
            tokdict[w] += 1
    for k, v in sorted(tokdict.items(), key=lambda x: -x[1])[:100]:
        print(k, v)
