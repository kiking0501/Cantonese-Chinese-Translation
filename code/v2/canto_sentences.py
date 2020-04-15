from dao_babel import read_clean_transcription_combine as read_babel_corpus
from dao_hkcancor import read_clean_transcription_combine as read_hkcancor_corpus
from dao_wiki import read_clean_wikipedia
from dao_wiki import _apply_cleansing as _wiki_apply_cleansing
from utilities_tokenize import load_jieba, DL
import jieba
from configuration import JIEBA_CANTO, SPM_CANTO, SPM_PATH
import sentencepiece as spm
import os
from bilingual_sentences import read_data as read_parallel_data


class CantoSentences(object):
    '''
        A generator for yielding monolinugal sentences (Cantonese)
        optional to use tokenizer (by Jieba or trained subword-tokenizer)

        - wiki: Wikimedia, babel: IARPA Babel, hkcancor: HKCancor
        - dev_data: Cantonese sentences extracted from the dev-set of the parallel corpus
        - see configuration.py for more details on the datasets
    '''
    jieba_dict = JIEBA_CANTO
    spm_model = SPM_CANTO + ".model"

    def __init__(self, tokenizer=None, wiki=True, babel=True, hkcancor=True,
                 dev_data=False):
        ''' tokenizer can be "jieba", "spm", or None
            if set to None, output sentence as string without tokenization
        '''
        if tokenizer == "jieba":
            load_jieba(self.jieba_dict)
            self.tokenize = self.jieba_tokenizer
        elif tokenizer == "spm":
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(os.path.join(SPM_PATH, "cantonese", self.spm_model))
            self.tokenize = self.spm_tokenizer
        else:
            self.tokenize = self.none_tokenizer

        self.tokenize_type = tokenizer

        self.use_wiki = wiki  # 427k
        self.use_babel = babel  # 91k
        self.use_hkcancor = hkcancor  # 9k
        self.use_dev_data = dev_data  # 8.5k

    def clean(self, sentence, corpus):
        if corpus == "wiki" and isinstance(sentence, str):
            sentence = _wiki_apply_cleansing(sentence) + "。"
            sentence = self.tokenize(sentence)

        if isinstance(sentence, str):
            sentence = self.tokenize(sentence)

        if self.tokenize_type == "jieba":
            sentence = [tok.lower() for tok in sentence if DL.lang(tok) != 'ENGLISH']
        return sentence

    def jieba_tokenizer(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

    def spm_tokenizer(self, sentence):
        return self.sp.EncodeAsPieces(sentence)

    def none_tokenizer(self, sentence):
        if isinstance(sentence, list):
            sentence = ''.join(sentence)
        return sentence

    def __iter__(self):
        if self.use_wiki:
            for sen in read_clean_wikipedia():
                yield self.clean(sen, "wiki")

        if self.use_babel:
            for sen in read_babel_corpus():
                yield self.clean(sen, "babel")

        if self.use_hkcancor:
            for sen in read_hkcancor_corpus():
                yield self.clean(sen, "hkcancor")

        if self.use_dev_data:
            for _, canto in read_parallel_data("dev"):
                yield self.clean(canto, "dev_data")


if __name__ == '__main__':
    from configuration import DATA_PATH

    DATA_SELECT = {
        'wiki': True,
        'babel': True,
        'hkcancor': True,
    }
    txt_path = os.path.join(DATA_PATH, "all_canto_sentences.txt")
    corpus = CantoSentences(**DATA_SELECT)

    with open(txt_path, "w") as f:
        for ind, sen in enumerate(corpus):
            f.write("%s\n" % sen)

            if ind % 1000 == 0:
                print("%d finished." % ind)
        print("Total: %d. %s saved." % (ind, f.name))

