from utilities_tokenize import load_jieba
from configuration import JIEBA_CANTO, SPM_CANTO, SPM_PATH
import jieba
import tensorflow as tf
import sentencepiece as spm
import numpy as np
import os


class Vocabulary(object):

    def __init__(self,
                 tokenizer="jieba",
                 jieba_dict=JIEBA_CANTO,
                 spm_model=SPM_CANTO + ".model"):

        ### tokenizer, change between jieba and spm
        if tokenizer == "jieba":
            load_jieba(jieba_dict)
            self.tokenize = self.jieba_tokenizer
        elif tokenizer == "spm":
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(os.path.join(SPM_PATH, "cantonese", spm_model))
            self.tokenize = self.spm_tokenizer
        else:
            self.tokenize = None

        self.jieba_dict = jieba_dict
        self.spm_model = spm_model
        self.tokenize_type = tokenizer

        self.unknown_word = "<unk>"
        self.sent_start = "<s>"
        self.sent_end = "</s>"
        self.pad_word = "<pad>"
        self.special_tokens = [
            self.sent_start, self.sent_end, self.pad_word, self.unknown_word
        ]
        self.word_counts, self.word2idx, self.idx2word, self.max_length = self.init_vocab()

    def init_vocab(self):
        word_counts = {}
        word2idx = {
            self.pad_word: 0,
            self.sent_start: 1,
            self.sent_end: 2,
            self.unknown_word: 3
        }
        idx2word = [k for k, v in sorted(word2idx.items(), key=lambda x: x[1])]
        return word_counts, word2idx, idx2word, 1

    def jieba_tokenizer(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

    def spm_tokenizer(self, sentence):
        return self.sp.EncodeAsPieces(sentence)

    def load_vocab(self, sentences):
        for sen in sentences:
            sentoks = self.tokenize(sen)
            for w in sentoks:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.idx2word)
                    self.idx2word.append(w)
                    self.word_counts[w] = 0
                self.word_counts[w] += 1
            self.max_length = max(len(sentoks), self.max_length)

        self.vocab_size = len(self.word2idx)

    def generate_sequences(self, sentoks_list, add_start_end=False):
        sequences = [
            [self.word2idx[w] if w in self.word2idx else self.word2idx[self.unknown_word]
             for w in sentoks]
            for sentoks in sentoks_list
        ]
        if add_start_end:
            sequences = [
                [self.word2idx[self.sent_start]] + seq + [self.word2idx[self.sent_end]]
                for seq in sequences
            ]
        return sequences

    def generate_pad_sequences(self, sentoks_list, add_start_end=False):
        sequences = self.generate_sequences(sentoks_list, add_start_end=add_start_end)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                               maxlen=self.max_length,
                                                               padding='post')
        return tensor


def read_embedding_vec(file_path, verbose=False):
    if verbose: print("Reading %s..." % file_path)
    vec_dict = {}
    cnt = 0
    with open(file_path) as f:
        line = f.readline()
        total, dim = [int(seg) for seg in line.split(" ")]

        line = f.readline()
        while line:
            l_list = line.split(" ")
            if l_list[0]:
                vec_dict[l_list[0]] = [float(seg) for seg in l_list[1:] if seg != '\n']
            else:  # special for space tokens
                vec_dict[" "] = [float(seg) for seg in l_list[2:] if seg != '\n']
            #if verbose: print("%d: %s" % (cnt, l_list[0]))
            cnt += 1
            line = f.readline()
        if verbose: print("Count(Total): %d (%d)" % (cnt, total))
    return vec_dict


def associate_vocab_with_embedding_weights(vocab_obj, embedding_dict):
    if not hasattr(vocab_obj, "vocab_size"):
        raise ValueError("You need to run vocab_obj.load_vocab(sentences) first to get vocab_size!")
    embedding_dim = 300
    matrix_shape = (vocab_obj.vocab_size, embedding_dim)
    embedding_matrix = np.random.normal(0, 1e-2, matrix_shape)

    not_found = 0
    for w, idx in vocab_obj.word2idx.items():
        if w in embedding_dict:
            embedding_matrix[idx] = embedding_dict[w]
        else:
            not_found += 1
    print("Found: %d/%d. (%.3f)" % (vocab_obj.vocab_size-not_found, vocab_obj.vocab_size, 1-not_found/vocab_obj.vocab_size))


if __name__ == '__main__':
    from bilingual_sentences import read_data
    from configuration import EMB_PATH

    dev_data = read_data("dev")
    train_data = read_data("train")
    valid_data = read_data("valid")
    test_data = read_data("test")

    vocab_dict = {
        type_: {
            'inp': Vocabulary(jieba_dict="dict.txt.big_trad"),
            'out': Vocabulary(jieba_dict="dict.txt.all_canto")
        } for type_ in ["dev", "test"]
    }

    for type_ in ["dev", "test"]:
        vocab_dict[type_]["inp"].load_vocab([stdch for stdch, _ in train_data])
        vocab_dict[type_]["out"].load_vocab([canto for _, canto in train_data])

    wv_jieba = "jieba-fasttext_babel_hkcancor_wiki"
    wv_dev = "jieba-fasttext_dev_data"

    emb_jieba = read_embedding_vec(os.path.join(EMB_PATH, "cantonese", wv_jieba+".vec"))
    matrix = associate_vocab_with_embedding_weights(vocab_dict["dev"]["out"], emb_jieba)

    emb_dev = read_embedding_vec(os.path.join(EMB_PATH, "cantonese", wv_dev+".vec"))
    matrix = associate_vocab_with_embedding_weights(vocab_dict["dev"]["out"], emb_dev)
