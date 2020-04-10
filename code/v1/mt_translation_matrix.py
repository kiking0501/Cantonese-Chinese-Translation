import pickle
import configuration as config
from dao import train_valid_split, char_cut
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gensim.models.wrappers import FastText
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

tf.set_random_seed(1)
np.random.seed(1)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = config.data_dir
output_dir = os.path.join(data_dir, "translation_matrix_model", "custom_canto_wiki")

trained_W_path = os.path.join(output_dir, "W.pkl")
# canto_lm = FastText.load_fasttext_format(os.path.join(data_dir, "embedding", "cantonese", "wiki.zh_yue.bin"))
canto_lm = KeyedVectors.load_word2vec_format(datapath(os.path.join(data_dir, "embedding", "cantonese", "custom_wiki.bin")), binary=True)


###### BLEU Testing #####

## Output Settings
TRANS_MATRIX_OUTPUT_FILE = "translatoin_matrix.output"
BLEU_OUTPUT_FILE = "translation_matrix.BLEU"

### Validation Settings
TEST_INPUT_TOKENIZED = "valid.stdch.sent.tok.dict.txt.big_trad"
TEST_GT_CHAR = "valid.canto.sent.tok.char"

## Test Settings
# TEST_INPUT_TOKENIZED = "test.stdch.sent.tok.dict.txt.big_trad"
# TEST_GT_CHAR = "test.canto.sent.tok.char"


def generate_output(stdch_emb, srcFile, trgFile, W=None, **kwargs):
    ''' map_type can either be "replace" or "regex"
            "replace": tokenize before mapping
            "regex" : direct mapping
    '''
    stdch_embedding = pickle.load(open(os.path.join(data_dir, "embedding", stdch_emb), "rb"))
    stdch_keys = stdch_embedding.keys()

    if W is None:
        W = pickle.load(open(trained_W_path, "rb"))['W']

    def _get_inference_word(st):
        if st in stdch_keys:
            choices = inference(np.array(stdch_embedding[st]), W, topN=1, **kwargs)
            if choices:
                print("%s -> %s" % (st, choices))
                return choices[0][0]
        return st

    with open(srcFile) as src_f:
        srcLines = src_f.readlines()

        with open(trgFile, "w") as trg_f:
            for sentence in srcLines:
                new_sentence = sentence
                new_sentok = sentence[:-1].split(" ")

                new_sentok = [_get_inference_word(st) for st in new_sentok]
                new_sentok = char_cut(new_sentok)
                new_sentence = " ".join(new_sentok)
                trg_f.write("%s\n" % new_sentence)
        print("%s saved." % trg_f.name)


###### Translation Matrix Model Training #######

SETTINGS = {
    "stdch": {
        "train": "train.stdch.emb",
        "valid": "valid.stdch.emb",
    },
    "canto": {
        "train": "train.canto.emb",
        "valid": "valid.canto.emb"
    }
}


def save_embeddings_with_frequencies(canto_emb, stdch_emb, canto_freq, stdch_freq, addl_dict):

    def read_freq_dict(dict_txt):
        with open(os.path.join(data_dir, "jieba_dict", dict_txt)) as f:
            freq_dict = {}
            for line in f.readlines():
                s = line.split(' ')
                freq_dict[s[0]] = int(s[1])
        return freq_dict

    canto_freq_dict, stdch_freq_dict = read_freq_dict(canto_freq), read_freq_dict(stdch_freq)
    canto_embedding = pickle.load(open(os.path.join(data_dir, "embedding", canto_emb), "rb"))
    stdch_embedding = pickle.load(open(os.path.join(data_dir, "embedding", stdch_emb), "rb"))

    # normalize different frequencies to 1000000-base
    normalize_base = 1e6
    canto_sum = sum(canto_freq_dict.values())
    canto_freq_dict = {k: int(v * (normalize_base / canto_sum))
                       for k, v in canto_freq_dict.items()}
    stdch_sum = sum(stdch_freq_dict.values())
    stdch_freq_dict = {k: int(v * (normalize_base / stdch_sum))
                       for k, v in stdch_freq_dict.items()}

    # Select only those with frequencies recorded
    canto_embedding = {k: v for k, v in canto_embedding.items()
                       if k in canto_freq_dict}
    stdch_embedding = {k: v for k, v in canto_embedding.items()
                       if k in stdch_freq_dict}
    canto_keys = set(canto_embedding.keys())
    stdch_keys = set(stdch_embedding.keys())

    # get words that appear in both embeddings, and their normalized frequencies
    inter_freq_dict = {k: canto_freq_dict[k] + stdch_freq_dict[k]
                       for k in canto_keys.intersection(stdch_keys)}
    print("Overlap Words: %d" % len(inter_freq_dict))
    # get words that appear in the bilingual dictionary in both embeddings
    bilingual_freq_dict = {}
    with open(os.path.join(data_dir, "static", addl_dict)) as f:
        lines = f.readlines()
        for l in lines:
            canto_w, stdch_list = l.split(' ')
            for stdch_w in stdch_list.split('/'):
                if canto_w in canto_keys and stdch_w in stdch_keys:
                    bilingual_freq_dict[(canto_w, stdch_w)] = canto_freq_dict[canto_w] + stdch_freq_dict[stdch_w]
    print("Dictionary-Mapped Words: %d" % len(bilingual_freq_dict))

    all_freq_dict = {}
    for k, v in inter_freq_dict.items():
        all_freq_dict[(k, k)] = v
    for (canto_w, stdch_w), v in bilingual_freq_dict.items():
        # HARD REMOVE
        if (stdch_w, stdch_w) in all_freq_dict:
            del all_freq_dict[(stdch_w, stdch_w)]
        all_freq_dict[(canto_w, stdch_w)] = v

    sorted_freq_list = sorted(all_freq_dict.items(), key=lambda x: (-x[1], len(x[0][0]), len(x[0][1])))
    with open(os.path.join(output_dir, "dict.txt.bilingual_map"), "w") as f:
        for (canto_w, stdch_w), v in sorted_freq_list:
            f.write("%s %s %s\n" % (canto_w, stdch_w, v))
        print("Total: %s." % len(sorted_freq_list))
        print("%s saved." % f.name)

    train_ratio = 0.8
    num_train = int(len(sorted_freq_list) * train_ratio)
    train_data, valid_data = sorted_freq_list[:num_train], sorted_freq_list[num_train:]

    for split, data in [
            ("train", train_data), ("valid", valid_data)]:

        stdch_f = open(os.path.join(output_dir, "%s.stdch.emb" % split), "wb")
        canto_f = open(os.path.join(output_dir, "%s.canto.emb" % split), "wb")
        pickle.dump({'data': [(stdch_w, stdch_embedding[stdch_w]) for (canto_w, stdch_w), c in data]}, stdch_f)  # a list of tuple (word, emb)
        pickle.dump({'data': [(canto_w, canto_embedding[canto_w]) for (canto_w, stdch_w), c in data]}, canto_f)  # a list of tuple (word, emb)
        stdch_f.close()
        canto_f.close()
        print("==== Split: %s has %d data." % (split, len(data)))
        print("%s saved." % stdch_f.name)
        print("%s saved." % canto_f.name)


def load_data(split, source='stdch', target='canto'):
    source_saved_pkl = pickle.load(
        open(os.path.join(output_dir, SETTINGS[source][split]), "rb"))
    X = np.array([t[1] for t in source_saved_pkl['data']])
    X_words = [t[0] for t in source_saved_pkl['data']]

    target_saved_pkl = pickle.load(
        open(os.path.join(output_dir, SETTINGS[target][split]), "rb"))
    Z = np.array([t[1] for t in target_saved_pkl['data']])
    Z_words = [t[0] for t in target_saved_pkl['data']]
    print("X.shape: %s, Z.shape: %s." % (X.shape, Z.shape))
    return (X, X_words), (Z, Z_words)


def SGDtrain(trainX, trainZ, validX, validZ,
             max_iter=5000, lr=1e-2, tol=1e-6,
             print_every=100):
    ''' find a transformation matrix W such that W * x(i) approx. z(i) by
            min_W  sum(i:n) ||W * x(i) - z(i)||^2
        using stochastic gradient descent
    '''
    tX, tX_words = trainX
    tZ, tZ_words = trainZ
    vX, vX_words = validX
    vZ, vZ_words = validZ

    n, d = tX.shape

    ## placeholder, variable declaring
    x_vec = tf.placeholder(shape=[d, 1], dtype=tf.float32)
    z_vec = tf.placeholder(shape=[d, 1], dtype=tf.float32)
    W = tf.get_variable(name="W", shape=[d, d], dtype=tf.float32)

    ## add functions
    z_pred = tf.matmul(W, x_vec)
    loss = tf.square(tf.norm(z_pred - z_vec))

    ## start
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_step = train_opt.minimize(loss)

    train_loss_list, valid_loss_list = [], []
    try:
        for t in range(max_iter):

            rand_ind = np.random.choice(n)

            sess.run(
                train_step,
                feed_dict={
                    x_vec: np.transpose(tX[rand_ind].reshape(1, d)),
                    z_vec: np.transpose(tZ[rand_ind].reshape(1, d))
                }
            )

            curr_W = sess.run(W)
            train_loss = np.square(
                np.linalg.norm(np.matmul(curr_W, np.transpose(tX)) - np.transpose(tZ))
            )
            valid_loss = np.square(
                np.linalg.norm(np.matmul(curr_W, np.transpose(vX)) - np.transpose(vZ))
            )
            if (t + 1) % print_every == 0:
                print("Step %5d/%5d: train-loss: %5.4f, valid-loss: %5.4f" %
                      (t + 1, max_iter, train_loss, valid_loss))
                check_accuracy(validX, validZ, W=curr_W, verbose=True)

            if (t > 0) and (abs(train_loss_list[-1] - train_loss) < tol):
                break

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
    except KeyboardInterrupt:
        pass

    final_w = sess.run(W)
    print("Final W Performance:")
    check_accuracy(validX, validZ, W=curr_W, verbose=False)
    return final_w


def check_accuracy(validX, validZ, W=None,
                   topN=5, threshold=0.2, verbose=True):
    vX, vX_words = validX
    vZ, vZ_words = validZ
    n, d = vX.shape

    if W is None:
        W = pickle.load(open(trained_W_path, "rb"))['W']

    match_count = 0
    for i, x_word, z_word in zip(range(n), vX_words, vZ_words):
        nearest = inference(vX[i], W=W, topN=topN, threshold=threshold)
        is_match = z_word in [t[0] for t in nearest]
        if is_match:
            match_count += 1
        if verbose:
            print("(%d): %s -> %s (%s)" % (
                i, z_word, '/'.join(['%s(%.2f)' % (t[0], t[1]) for t in nearest]), is_match
            ))
    print("Validation Accuracy (topN=%d, TH=%.2f): %d / %d (%.4f)" %
          (topN, threshold, match_count, n, match_count / n))
    return


def inference(stdch_w_vec, W=None,
              topN=1, threshold=0.6, verbose=False):
    if W is None:
        W = pickle.load(open(trained_W_path, "rb"))['W']
    d = W.shape[0]
    pred = np.matmul(W, np.transpose(stdch_w_vec.reshape(1, d)))[:, 0]
    nearest = [t for t in canto_lm.similar_by_vector(pred, topn=topN)
               if t[1] >= threshold]

    return nearest


if __name__ == '__main__':

    ### Check Internal Accuracy ###
    THRESHOLD = 0.2


    # if not all([os.path.exists(os.path.join(output_dir, SETTINGS[src][split]))
    #            for split in ["train", "valid"] for src in ["canto", "stdch"]]):
    #     save_embeddings_with_frequencies(
    #         "canto_wiki.pkl", "stdch_wiki.pkl", "dict.txt.pycanto", "dict.txt.big_trad",
    #         "canto2stdch_full.dict"
    #     )
    if not all([os.path.exists(os.path.join(output_dir, SETTINGS[src][split]))
               for split in ["train", "valid"] for src in ["canto", "stdch"]]):
        save_embeddings_with_frequencies(
            "custom_canto_wiki.pkl", "stdch_wiki.pkl", "dict.txt.pycanto", "dict.txt.big_trad",
            "canto2stdch_full.dict"
        )

    trainX, trainZ = load_data("train")
    validX, validZ = load_data("valid")

    if not os.path.exists(trained_W_path):
        W = SGDtrain(trainX, trainZ, validX, validZ, lr=1e-3, max_iter=5000)  #1e-2 for original
        pickle.dump({"W": W}, open(trained_W_path, "wb"))
        print("%s saved." % trained_W_path)

    check_accuracy(validX, validZ, topN=1, threshold=THRESHOLD)

    ### Check BLEU on translation task ###
    testInput = os.path.join(config.data_dir, TEST_INPUT_TOKENIZED)
    testGT = os.path.join(config.data_dir, TEST_GT_CHAR)
    TransMatrixOutputFile = os.path.join(output_dir, TRANS_MATRIX_OUTPUT_FILE)

    generate_output(
        "stdch_wiki.pkl", testInput, TransMatrixOutputFile, threshold=THRESHOLD, verbose=True
    )

    print("perl multi-bleu.perl -lc " + testGT + " < " + TransMatrixOutputFile)
    BLEUOutput = os.popen("perl multi-bleu.perl -lc " + testGT + " < " + TransMatrixOutputFile).read()
    with open(os.path.join(output_dir, BLEU_OUTPUT_FILE), "w") as f:
        print(BLEUOutput)
        f.write(BLEUOutput)
