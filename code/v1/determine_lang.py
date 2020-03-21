''' Personal Use only '''

import re
import os
from collections import defaultdict

data_dir = "../../data/"


UNICODE_BLOCK_PATH = os.path.join(data_dir, 'static', 'UnicodeBlocks.txt')
LANGUAGE_BLOCKS = {
    'CHINESE': [
        'CJK Radicals Supplement',
        'Kangxi Radicals',
        'CJK Unified Ideographs Extension A',
        'CJK Unified Ideographs',
        'CJK Compatibility Ideographs',
    ],
    'ENGLISH': [
        'Basic Latin',
    ],
    'SYMBOLS': [
        'General Punctuation',
        'Halfwidth and Fullwidth Forms',
    ]
}


def _find_blocks():
    blocks = []
    text = open(UNICODE_BLOCK_PATH).read()
    pattern = re.compile(r'([0-9A-F]+)\.\.([0-9A-F]+);\ (\S.*\S)')
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            start, end, name = m.groups()
            blocks.append((int(start, 16), int(end, 16), name))
    return blocks


_blocks = _find_blocks()
_lang_re = {
    lang: (u'[%s]' % u''.join(
        u'%s-%s' % (chr([b for b in _blocks if b[2] == block][0][0]),
                    chr([b for b in _blocks if b[2] == block][0][1]))
        for block in blocks))
    for lang, blocks in LANGUAGE_BLOCKS.items()
}

_lang_compile_re = {
    lang: re.compile(s)
    for lang, s in _lang_re.items()
}

_lang_all_re = {
    lang: re.compile(s + '+$')
    for lang, s in _lang_re.items()
}


def contains_language(string, language):
    ''' faster function to test if it contains any of a language '''
    return _lang_compile_re[language].search(string) is not None


def count_language(string):
    count_map = defaultdict(int)
    other = len(string)
    for lang, lang_re in _lang_compile_re.items():
        if contains_language(string, lang):
            count_map[lang] = len(lang_re.findall(string))
            other -= count_map[lang]
    if other > 0:
        count_map['OTHER'] = other
    return count_map


class DetermineLanguage(object):
    ''' detemine whether a sentence is CHINESE / ENGLISH / MIXED / OTHER'''

    def __init__(self):
        self.re_alpha = re.compile('[a-zA-Z]')

    def lang(self, string):
        lang_count = count_language(string)
        if 'CHINESE' in lang_count or 'ENGLISH' in lang_count:
            containEngChar = self.re_alpha.search(string)
            if containEngChar:
                #check if contain CHI char
                if lang_count['CHINESE']:
                    return 'MIXED'
                else:
                    return 'ENGLISH'
            else:
                return 'CHINESE'
        elif lang_count:
            return max(lang_count, key=lang_count.get)
