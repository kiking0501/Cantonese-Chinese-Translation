import os
from configuration import (
    TATOEBA_PATH, TATOEBA_STDCH_PATH, TATOEBA_CANTO_PATH
)
from opencc import OpenCC
import re


if not os.path.exists(os.path.join(TATOEBA_STDCH_PATH)):
    raise IOError("You need to download cmn-eng.zip (Chinese (Mandarin) - English) from\n"
                  " https://www.manythings.org/anki/\n"
                  " extract and place the compete cmn-eng folder\n"
                  " under %s!" % TATOEBA_PATH)

if not os.path.exists(os.path.join(TATOEBA_CANTO_PATH)):
    raise IOError("You need to download yue-eng.zip (Cantonese - English)from\n"
                  " https://www.manythings.org/anki/\n"
                  " extract and place the compete cmn-eng folder\n"
                  " under %s!" % TATOEBA_PATH)

s2t = OpenCC('s2t')


def read_eng_translation(lang):
    if lang == "stdch":
        path = TATOEBA_STDCH_PATH
    elif lang == "canto":
        path = TATOEBA_CANTO_PATH

    eng_dict = {}
    with open(path) as f:
        for l in f.readlines():
            line = l.split('\t')
            eng, trans = line[0], line[1]
            if eng in eng_dict:
                eng_dict[eng].append(trans)
            else:
                eng_dict[eng] = [trans]
    return eng_dict


def read_all_data(apply_cleansing=True):
    '''' {English sentence:
            {'stdch': [sent1, sent2...], 'canto: [sent1, sent2...]}}
    '''
    eng_stdch = read_eng_translation("stdch")
    eng_canto = read_eng_translation("canto")

    eng_both = {}
    for eng in set(eng_stdch.keys()).intersection(eng_canto.keys()):

        eng_both[eng] = {
            'stdch': eng_stdch[eng],
            'canto': eng_canto[eng]
        }

        if apply_cleansing:
            for lang in ('stdch', 'canto'):
                if lang == 'stdch':
                    eng_both[eng][lang] = [s2t.convert(s) for s in eng_both[eng][lang]]
                eng_both[eng][lang] = [re.sub("\"|”|“|「|」", "", s) for s in eng_both[eng][lang]]
    return eng_both


def read_data(apply_cleansing=True):
    eng_both = read_all_data(apply_cleansing=apply_cleansing)
    data = [(v['stdch'][0], v['canto'][0]) for k, v in sorted(eng_both.items())]
    return data
