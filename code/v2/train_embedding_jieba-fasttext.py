from gensim.models.fasttext import FastText  # Word2Vec
from canto_sentences import CantoSentences
from stdch_sentences import StdchSentences
import time
import logging
from configuration import DATA_PATH
import os
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


DATA_SELECT = {
    ### General purpose
    # 'wiki': True,
    # 'babel': True,
    # 'hkcancor': True,
    # 'dev_data': False,

    ### Task specific
    'wiki': False,
    'babel': False,
    'hkcancor': False,
    'dev_data': True,
}

# Default config from Bojanowski, 2017
PARAM = {
    'min_n': 3,  # n-grams information
    'max_n': 6,
    'size': 300,  # dimension
    'negative': 5,  # for each positive, sample 5 negativs at random
    'sample': 1e-4,  # rejetion threshold
    'alpha': 0.05,  # step size
    'iter': 10,

    # ### General purpose
    # 'min_count': 5,  # dictionary count
    ### Task specific
    'min_count': 1,
}

EMBEDDING_PATH = os.path.join(DATA_PATH, "embedding", "standard_chinese")


def save_vec(model, model_name):
    with open(os.path.join(EMBEDDING_PATH, "%s.vec" % model_name), "w") as f:
        f.write("%s %s\n" % (len(model.wv.index2word), PARAM['size']))
        for word, vec in zip(model.wv.index2word, model.wv.vectors):
            f.write(' '.join([word] + ['%.6f' % x for x in vec]) + '\n')
        print("%s saved." % f)

    with open(os.path.join(EMBEDDING_PATH, "%s.index2word" % model_name), "w") as f:
        for word in sorted(model.wv.index2word, key=lambda x: -len(x)):
            f.write(word + '\n')
        print("%s saved." % f)


model_name = '_'.join(['jieba-fasttext'] + [k for k, v in sorted(DATA_SELECT.items()) if v])


print("Start Training: %s" % model_name)
st = time.time()

corpus = StdchSentences(tokenizer="jieba", **DATA_SELECT)
model = FastText(corpus, **PARAM)

print("Training time: %s mins." % ((time.time() - st) / 60))


model.save(os.path.join(EMBEDDING_PATH, "%s.model" % model_name))
model.wv.save_word2vec_format(os.path.join(EMBEDDING_PATH, '%s.bin' % model_name), binary=True)
save_vec(model, model_name)

'''
Load Model:

from gensim.models.fasttext import FastText

model_path = '../../data/embedding/cantonese/custom_babel_hkcancor_wiki.model'
saved_model = FastText.load(model_path)
print(len(saved_model.wv.index2model))

'''
