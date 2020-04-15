from gensim.models import Word2Vec
from canto_sentences import CantoSentences
import time
import logging
from configuration import DATA_PATH
import os


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


DATA_SELECT = {
    'wiki': True,
    'babel': True,
    'hkcancor': True,
}

# Default config from Mikolov, 2013
PARAM = {
    'size': 300,  # dimension
    'negative': 5,  # for each positive, sample 5 negativs at random
    'sample': 1e-4,  # rejetion threshold
    'min_count': 5,  # dictionary count
    'alpha': 0.05,  # step size
    'iter': 10,
}

EMBEDDING_PATH = os.path.join(DATA_PATH, "embedding", "cantonese")


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


model_name = '_'.join(['spm-word2vec'] + [k for k, v in sorted(DATA_SELECT.items()) if v])


print("Start Training: %s" % model_name)
st = time.time()

corpus = CantoSentences(tokenizer="spm", **DATA_SELECT)
model = Word2Vec(corpus, **PARAM)

print("Training time: %s mins." % ((time.time() - st) / 60))


model.save(os.path.join(EMBEDDING_PATH, "%s.model" % model_name))
model.wv.save_word2vec_format(os.path.join(EMBEDDING_PATH, '%s.bin' % model_name), binary=True)
save_vec(model, model_name)

'''
Load Model:

from gensim.models import Word2Vec

model_path = '../../data/embedding/cantonese/custom_babel_hkcancor_wiki.model'
saved_model = Word2Vec.load(model_path)
print(len(saved_model.wv.index2model))

'''
