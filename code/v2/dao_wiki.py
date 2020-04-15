import os
from collections import defaultdict
from configuration import WIKI_PATH, WIKI_ORI_PATH, DATA_PATH, JIEBA_DICT_PATH
from determine_lang import DetermineLanguage
import json
import re
from utilities_tokenize import DL, load_jieba
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


SAVE_DIR = os.path.join(WIKI_PATH, "clean")
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def read_wiki(file_name):
    data = []
    with open(os.path.join(WIKI_ORI_PATH, file_name), 'r') as f:
        for json_obj in f:
            data.append(json.loads(json_obj))
    return data


def _apply_cleansing(line):
    content = re.findall('\（.*?\）', line)
    for c in content:
        if DL.lang(c) in ['SYMBOLS', 'ENGLISH']:
            line = line.replace(c, '')
    return line


def save_clean_wiki(file_name, tokenize=False, apply_cleansing=False, verbose=True):
    json_list = read_wiki(file_name)
    for json_obj in json_list:
        output_path = os.path.join(SAVE_DIR, "%s_%s" % (json_obj['id'], json_obj['title'].replace('/', '-').strip()))
        with open(output_path, "w") as f:
            for line in json_obj['text'].split('\n'):
                if apply_cleansing:
                    line = _apply_cleansing(line)

                for l in line.split('。'):
                    if l:
                        if tokenize:
                            f.write(' '.join(jieba.cut(l, cut_all=False)) + ' 。\n')
                        else:
                            f.write(l + '。\n')
        print("%s saved." % output_path)


def read_clean_wiki(file_code, tokenize=False, apply_cleansing=False):
    with open(os.path.join(SAVE_DIR, file_code)) as f:
        if tokenize:
            sentences = [l.strip().split(' ') for l in f]
        else:
            sentences = [_apply_cleansing(l).strip() if apply_cleansing else l.strip() for l in f]
        return sentences


def save_clean_wikipedia(output_file="wiki_yue_overview.csv", tokenize=False, apply_cleansing=False, verbose=True):
    if tokenize:
        load_jieba()

    for (_, _, filenames) in sorted(os.walk(WIKI_ORI_PATH)):
        for file_name in sorted(filenames):
            save_clean_wiki(file_name, tokenize=tokenize, apply_cleansing=apply_cleansing)

    total = 0
    with open(os.path.join(WIKI_PATH, output_file), "w") as f:
        file_codes = []
        for (_, _, filenames) in os.walk(SAVE_DIR):
            file_codes.extend(filenames)
        for ind, code in enumerate(sorted(file_codes)):
            f.write('%s,%s\n' % (ind, code))
        total += len(file_codes)
    if verbose:
        print("Total: %d. %s saved." % (total, output_file))


def read_clean_wikipedia(overview_csv="wiki_yue_overview.csv", tokenize=False):
    with open(os.path.join(WIKI_PATH, overview_csv)) as f:
        for line in f.readlines():
            # print(line)
            _, file_code = line.partition(',')[0], line.partition(',')[2].strip()
            if file_code.startswith('wiki'):
                continue
            # print("Reading %s.." % file_code)
            for sen in read_clean_wiki(file_code, tokenize=tokenize):
                yield sen

if __name__ == '__main__':
    '''
        Run in terminal:

        $ cd ./data/dl-Wikipedia-YUE
        $ OUTPUT_DIR=20200401
        $ DUMP_FILE=zh_yuewiki-20200401-pages-articles-multistream.xml.bz2
        $ wikiextractor-master/WikiExtractor.py -o $OUTPUT_DIR --json -b 500K $DUMP_FILE
        $ mv $OUTPUT_DIR/AA $OUTPUT_DIR/json

    '''
    save_clean_wikipedia()

    tokdict = defaultdict(int)
    for sen in read_clean_wikipedia():
        for w in sen:
            tokdict[w] += 1
    for k, v in sorted(tokdict.items(), key=lambda x: -x[1])[:100]:
        print(k, v)
