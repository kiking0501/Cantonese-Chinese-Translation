import configuration as config
import pickle
from collections import defaultdict
import os


class EmbeddingsCoverage(object):

    DATASETS = {
        ('canto', 'train', 'char'): "train.canto.sent.tok.char",
        ('canto', 'valid', 'char'): "valid.canto.sent.tok.char",

        ('canto', 'train', 'pycanto'): "train.canto.sent.tok.dict.txt.pycanto",
        ('canto', 'valid', 'pycanto'): "valid.canto.sent.tok.dict.txt.pycanto",

        ('stdch', 'train', 'char'): "train.stdch.sent.tok.char",
        ('stdch', 'valid', 'char'): "valid.stdch.sent.tok.char",

        ('stdch', 'train', 'big_trad'): "train.stdch.sent.tok.dict.txt.big_trad",
        ('stdch', 'valid', 'big_trad'): "valid.stdch.sent.tok.dict.txt.big_trad",
    }

    EMBEDDINGS = {
        ('canto', 'fasttext'): "canto.pkl",
        ('stdch', 'fasttext'): "stdch.pkl",
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
                for k, v in pickle.load(
                        open(os.path.join(config.data_dir, 'embedding', emb_file), "rb")).items():
                    emb_counter[k] += v

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
        embedding_param=('canto', 'fasttext'))

    EmbeddingsCoverage.check(
        dataset_param=('canto', 'ALL', 'char'),
        embedding_param=('canto', 'fasttext'))

    EmbeddingsCoverage.check(
        dataset_param=('stdch', 'ALL', 'big_trad'),
        embedding_param=('stdch', 'fasttext'))

    EmbeddingsCoverage.check(
        dataset_param=('stdch', 'ALL', 'char'),
        embedding_param=('stdch', 'fasttext'))
