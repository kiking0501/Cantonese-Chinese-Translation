from dao_movie import read_data as read_movie
from dao_tatoeba import read_data as read_tatoeba
from dao_UD import read_data as read_UD
from utilities_tokenize import load_jieba, DL
import jieba
from configuration import JIEBA_CANTO, BI_SENTENCES, DATA_PATH
import os


class StdchCantoSentences(object):
    '''
        A generator for yielding parallel sentences (Standard-Chinese, Cantonese)

        - movie: Movie collections
        - tatoeba: The Tatoeba Project
        - UD: the Catonese-Mandarin Parallel Dependcy Treebank

        is_dev=True: 8508 pairs (without cleansing: 8635 pairs)
        is_dev=False: 955 pairs (without cleansing: 1004 pairs)

        see configuration.py for more details on the datasets

    '''
    jieba_dict = JIEBA_CANTO

    def __init__(self, is_dev=None, movie=True, tatoeba=True, UD=False, apply_cleansing=True):
        self.is_dev = is_dev
        self.use_movie = movie
        self.use_tatoeba = tatoeba
        self.use_UD = UD
        self.apply_cleansing = apply_cleansing

        ### override
        if self.is_dev:
            self.use_movie = True
            self.use_tatoeba = True
            self.use_UD = False
        elif self.is_dev is False:
            self.use_movie = False
            self.use_tatoeba = False
            self.use_UD = True

        load_jieba(self.jieba_dict)

    def clean(self, sentence_pair, corpus):
        if not self.apply_cleansing:
            return sentence_pair

        stdch, canto = sentence_pair

        if corpus == "movie":
            stdch, canto = stdch.replace(" ", ","), canto.replace(" ", ",")
        stdch_tok = [tok.lower() for tok in self.jieba_tokenizer(stdch)]
        canto_tok = [tok.lower() for tok in self.jieba_tokenizer(canto)]

        # discard all data with English tokens
        for tok in stdch_tok + canto_tok:
            if DL.lang(tok) == "ENGLISH":
                return None

        stdch, canto = ''.join(stdch_tok).strip(), ''.join(canto_tok).strip()
        return (stdch, canto)

    def jieba_tokenizer(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

    def __iter__(self):
        if self.use_movie:
            for sen_pair in read_movie():
                sen_pair = self.clean(sen_pair, "movie")
                if sen_pair:
                    yield sen_pair
        if self.use_tatoeba:
            for sen_pair in read_tatoeba():
                sen_pair = self.clean(sen_pair, "tatoeba")
                if sen_pair:
                    yield sen_pair
        if self.use_UD:
            for sen_pair in read_UD():
                sen_pair = self.clean(sen_pair, "UD")
                if sen_pair:
                    yield sen_pair


def read_data(type_):
    with open(os.path.join(DATA_PATH, "static", BI_SENTENCES[type_])) as f:
        data = []
        for ind, l in enumerate(f.readlines()):
            stdch, canto = l.strip().split('\t')
            data.append((stdch, canto))
        return data


if __name__ == '__main__':
    from dao_movie import train_valid_split
    import os
    from configuration import DATA_PATH

    dev_data = list(StdchCantoSentences(is_dev=True))
    test_data = list(StdchCantoSentences(is_dev=False))

    ## Train: 7657
    ## Valid: 851
    ## Test: 955
    train_data, valid_data = train_valid_split(dev_data, train_ratio=0.9, test_num=None)

    for pairs, type_ in [(dev_data, "dev"), (train_data, "train"), (valid_data, "valid"), (test_data, "test")]:
        with open(os.path.join(DATA_PATH, "static", "bilingual_sentences_%s.txt" % type_), "w") as f:
            for ind, (stdch, canto) in enumerate(pairs):
                f.write("%s\t%s\n" % (stdch, canto))
            print("Total %d. %s saved." % (ind+1, f.name))
