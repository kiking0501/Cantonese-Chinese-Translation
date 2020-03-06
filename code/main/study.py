import configuration as config
import pickle
from collections import defaultdict
import os
from dao import save_sen2tok

class EmbeddingsCoverage(object):

    DATASETS = {
        ('canto', 'train', 'char'): "train.canto.sent.tok.char",
        ('canto', 'valid', 'char'): "valid.canto.sent.tok.char",

        ('canto', 'train', 'big_trad'): "train.stdch.sent.tok.dict.txt.big_trad",
        ('canto', 'valid', 'big_trad'): "valid.stdch.sent.tok.dict.txt.big_trad",

        ('canto', 'train', 'big'): "train.stdch.sent.tok.dict.txt.big",
        ('canto', 'valid', 'big'): "valid.stdch.sent.tok.dict.txt.big",

        ('canto', 'train', 'pycanto'): "train.canto.sent.tok.dict.txt.pycanto",
        ('canto', 'valid', 'pycanto'): "valid.canto.sent.tok.dict.txt.pycanto",

        ('canto', 'train', 'pycanto-big_trad'): "train.canto.sent.tok.dict.txt.pycanto-big_trad",
        ('canto', 'valid', 'pycanto-big_trad'): "valid.canto.sent.tok.dict.txt.pycanto-big_trad",

        ('canto', 'train', 'pycanto-canto_wiki'): "train.canto.sent.tok.dict.txt.pycanto-canto_wiki",
        ('canto', 'valid', 'pycanto-canto_wiki'): "valid.canto.sent.tok.dict.txt.pycanto-canto_wiki",

        ('canto', 'train', 'pycanto-stdch_wiki'): "train.canto.sent.tok.dict.txt.pycanto-stdch_wiki",
        ('canto', 'valid', 'pycanto-stdch_wiki'): "valid.canto.sent.tok.dict.txt.pycanto-stdch_wiki",

        ('stdch', 'train', 'char'): "train.stdch.sent.tok.char",
        ('stdch', 'valid', 'char'): "valid.stdch.sent.tok.char",

        ('stdch', 'train', 'big_trad'): "train.stdch.sent.tok.dict.txt.big_trad",
        ('stdch', 'valid', 'big_trad'): "valid.stdch.sent.tok.dict.txt.big_trad",

        ('stdch', 'train', 'big'): "train.stdch.sent.tok.dict.txt.big",
        ('stdch', 'valid', 'big'): "valid.stdch.sent.tok.dict.txt.big",
    }

    EMBEDDINGS = {
        ('canto', 'fasttext'): "canto_wiki.pkl",
        ('stdch', 'fasttext'): "stdch_wiki.pkl",
    }

    @classmethod
    def count_tokens(cls, tok_file):
        token_counter = defaultdict(int)

        with open(tok_file, "r") as f:
            sen_list = f.readlines()
            for sen in sen_list:
                for token in sen[:-1].split(" "):
                    token_counter[token] += 1
        return token_counter

    @classmethod
    def check(cls, dataset_param, embedding_param):
        ''' avaialbe param: see class init variables
            use 'ALL' as wildcard
        '''
        def _satisfy_param(input_param, cls_param):
            satisfy = True
            for ip, cp in zip(input_param, cls_param):
                if (ip != 'ALL') and (ip != cp):
                    satisfy = False
                    break
            return satisfy

        dataset_list = [
            v for k, v in cls.DATASETS.items()
            if (_satisfy_param(dataset_param, k))
        ]
        embedding_list = [
            v for k, v in cls.EMBEDDINGS.items()
            if (_satisfy_param(embedding_param, k))
        ]
        print("Params: %s, %s" % (dataset_param, embedding_param))

        print("=== Datasets(Tokenized Files): ")
        for li in dataset_list:
            print(li)
            if not os.path.exists(os.path.join(config.data_dir, li)):
                sen, dict_txt = li.split('.tok.')
                save_sen2tok(sen, dict_txt)
        print("=== Embedding:")
        for li in embedding_list:
            print(li)

        token_counter = defaultdict(int)
        for tok_file in dataset_list:
            for k, v in cls.count_tokens(os.path.join(config.data_dir, tok_file)).items():
                token_counter[k] += v

        if len(embedding_list) == 1:
            emb_counter = pickle.load(
                open(os.path.join(config.data_dir, 'embedding', embedding_list[0]), "rb"))
        else:
            emb_counter = defaultdict(int)
            for emb_file in embedding_list:
                for k in pickle.load(
                        open(os.path.join(config.data_dir, 'embedding', emb_file), "rb")).keys():
                    emb_counter[k] += 1

        inter_tokens = set(emb_counter.keys()).intersection(token_counter.keys())
        print ("=== Embeddings Coverage: %d / %d (%.4f)" %
               (len(inter_tokens), len(token_counter), len(inter_tokens) / len(token_counter)))
        print ("=== Embeddings Proportion: %d / %d (%.4f)" %
               (len(inter_tokens), len(emb_counter), len(inter_tokens) / len(emb_counter)))
        print("\n")
        return inter_tokens

if __name__ == '__main__':
    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto'),
        embedding_param=('canto', 'fasttext'))  # 0.6591 / 0.0140

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto-canto_wiki'),
        embedding_param=('canto', 'fasttext'))  # 0.6873 / 0.0146

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto-stdch_wiki'),
        embedding_param=('canto', 'fasttext'))  # 0.6765 / 0.0143

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto-big_trad'),
        embedding_param=('canto', 'fasttext'))  # 0.6527 / 0.0136

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto'),
        embedding_param=('ALL', 'fasttext'))  # 0.6906 / 0.0023

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'pycanto'),
        embedding_param=('stdch', 'fasttext'))  # 0.6896 / 0.0023

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'char'),
        embedding_param=('canto', 'fasttext'))  # 0.9636  / 0.0149

    EmbeddingsCoverage.check(
        dataset_param=('stdch', 'ALL', 'big_trad'),
        embedding_param=('stdch', 'fasttext'))  # 0.8093 / 0.0025

    EmbeddingsCoverage.check(
        dataset_param=('stdch', 'ALL', 'char'),
        embedding_param=('stdch', 'fasttext'))  # 0.9901 / 0.0023
