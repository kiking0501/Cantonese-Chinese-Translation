import configuration as config
import jieba
import dao
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np


def get_sen2tok(filename, dict_txt):
    ''' Modification:
            Tokenization with Jieba + Customized Dictionary
    '''
    tok_file = os.path.join(config.data_dir, filename + ".tok." + dict_txt)
    #if not os.path.exists(tok_file):
    dao.save_sen2tok(filename, dict_txt)

    with open(tok_file, "r") as f:
        sen_list = f.readlines()
    # num = int(sen_list[0])
    # sen_list = sen_list[1:]
    return [sen[:-1].split(" ") for sen in sen_list]  # avoid "\n" at the end


@config.CACHE("loadVocab.pkl")
def _loadVocab(inp_sen_file, out_sen_file):

    print("="*5, " loadVocab: %s / %s" % (inp_sen_file, out_sen_file))

    inputs = get_sen2tok(inp_sen_file, PreProcessing.INP_TOK)
    outputs = get_sen2tok(out_sen_file, PreProcessing.OUT_TOK)

    word_counters, word_to_idx, word_to_idx_ctr, idx_to_word = PreProcessing.initVocabItems()

    texts = inputs
    for text in texts:
        for token in text:
            if token not in word_to_idx:
                word_to_idx[token] = word_to_idx_ctr
                idx_to_word[word_to_idx_ctr] = token
                word_to_idx_ctr += 1
                word_counters[token] = 0
            word_counters[token] += 1
    texts = outputs
    for text in texts:
        for token in text:
            if token not in word_to_idx:
                word_to_idx[token] = word_to_idx_ctr
                idx_to_word[word_to_idx_ctr] = token
                word_to_idx_ctr += 1
                word_counters[token] = 0
            word_counters[token] += 1

    vocab_size = len(word_to_idx)
    return word_to_idx, idx_to_word, vocab_size, word_to_idx_ctr, word_counters


class PreProcessing:

    INP_SRC = {
        "train": "train.stdch.sent",
        "valid": "valid.stdch.sent",
        "test": "test.stdch.sent"
    }

    OUT_SRC = {
        "train": "train.canto.sent",
        "valid": "valid.canto.sent",
        "test": "test.canto.sent"
    }

    OUT_TOK = "dict.txt.pycanto-stdch_wiki"
    INP_TOK = "dict.txt.big_trad"
    # OUT_TOK = "char"
    # INP_TOK = "char"

    def __init__(self):
        self.sent_start, self.sent_end, self.pad_word, self.unknown_word = self._initVocabItems()
        self.special_tokens = [self.sent_start, self.sent_end, self.pad_word, self.unknown_word]
        self.word_counters, self.word_to_idx, self.word_to_idx_ctr, self.idx_to_word = self.initVocabItems()

    @staticmethod
    def _initVocabItems():
        unknown_word = "UNK".lower()
        sent_start = "SENTSTART".lower()
        sent_end = "SENTEND".lower()
        pad_word = "PADWORD".lower()
        special_tokens = [sent_start, sent_end, pad_word, unknown_word]
        return special_tokens

    @staticmethod
    def initVocabItems():
        sent_start, sent_end, pad_word, unknown_word = PreProcessing._initVocabItems()
        word_counters = {}
        word_to_idx = {}
        word_to_idx_ctr = 0
        idx_to_word = {}

        word_to_idx[pad_word] = word_to_idx_ctr  # 0 is for padword
        idx_to_word[word_to_idx_ctr] = pad_word
        word_counters[pad_word] = 1
        word_to_idx_ctr += 1

        word_to_idx[sent_start] = word_to_idx_ctr  # 1 is for SENTSTART
        word_counters[sent_start] = 1
        idx_to_word[word_to_idx_ctr] = sent_start
        word_to_idx_ctr += 1

        word_to_idx[sent_end] = word_to_idx_ctr  # 2 is for SENTEND
        word_counters[sent_end] = 1
        idx_to_word[word_to_idx_ctr] = sent_end
        word_to_idx_ctr += 1

        word_to_idx[unknown_word] = word_to_idx_ctr  # 3 is for UNK
        word_counters[unknown_word] = 1
        idx_to_word[word_to_idx_ctr] = unknown_word
        word_to_idx_ctr += 1

        return word_counters, word_to_idx, word_to_idx_ctr, idx_to_word

    def preprocess(self, filename, dict_txt):
        ''' Modification:
                Tokenization with Jieba + Customized Dictionary
        '''
        return get_sen2tok(filename, dict_txt)

    def loadVocab(self, split):
        print("="*5, " loadData: split = ", split)
        self.word_to_idx, self.idx_to_word, self.vocab_size, self.word_to_idx_ctr, self.word_counters \
            = _loadVocab(self.INP_SRC[split], self.OUT_SRC[split])
        return self.word_to_idx, self.idx_to_word, self.vocab_size, self.word_to_idx_ctr, self.word_counters

    def pruneVocab(self, max_vocab_size):
        word_to_idx = self.word_to_idx
        idx_to_word = self.idx_to_word
        word_to_idx_ctr = self.word_to_idx_ctr
        word_counters = self.word_counters

        tmp_word_counters, tmp_word_to_idx, tmp_word_to_idx_ctr, tmp_idx_to_word = self.initVocabItems()
        print ("** ", tmp_idx_to_word[1])

        print ("vocab size before pruning = ", len(word_to_idx))
        top_items = sorted( word_counters.items(), key=lambda x: -x[1])[:max_vocab_size]
        for token_count in top_items:
            token = token_count[0]
            if token in self.special_tokens:
                continue
            tmp_word_to_idx[token] = tmp_word_to_idx_ctr
            tmp_idx_to_word[tmp_word_to_idx_ctr] = token
            tmp_word_to_idx_ctr += 1

        self.word_to_idx = tmp_word_to_idx
        self.idx_to_word = tmp_idx_to_word
        self.vocab_size = len(tmp_word_to_idx)
        self.word_to_idx_ctr = tmp_word_to_idx_ctr
        print ("vocab size after pruning = ", self.vocab_size)

    def loadData(self, split):
        print("="*5, " loadData: split = ", split)
        inputs = self.preprocess(self.INP_SRC[split], self.INP_TOK)
        outputs = self.preprocess(self.OUT_SRC[split], self.OUT_TOK)

        word_to_idx = self.word_to_idx
        idx_to_word = self.idx_to_word
        word_to_idx_ctr = self.word_to_idx_ctr

        # generate sequences
        sequences_input = []
        sequences_output = []

        texts = inputs
        for text in texts:
            tmp = [word_to_idx[self.sent_start]]
            for token in text:
                if token not in word_to_idx:
                    tmp.append(word_to_idx[self.unknown_word])
                else:
                    tmp.append(word_to_idx[token])
            tmp.append(word_to_idx[self.sent_end])
            sequences_input.append(tmp)

        texts = outputs
        for text in texts:
            tmp = [word_to_idx[self.sent_start]]
            for token in text:
                if token not in word_to_idx:
                    tmp.append(word_to_idx[self.unknown_word])
                else:
                    tmp.append(word_to_idx[token])
            tmp.append(word_to_idx[self.sent_end])
            sequences_output.append(tmp)

        # pad sequences
        # sequences_input, sequences_output = padAsPerBuckets(sequences_input, sequences_output)
        sequences_input = pad_sequences(
            sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
        sequences_output = pad_sequences(
            sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')

        print("Printing few sample sequences... ")
        print(sequences_input[0],":", self.fromIdxSeqToVocabSeq(sequences_input[0]),
              "---",
              sequences_output[0], ":", self.fromIdxSeqToVocabSeq(sequences_output[0]))
        print ("=" * 5)

        return sequences_input, sequences_output

    def fromIdxSeqToVocabSeq(self, seq):
        return [self.idx_to_word[x] for x in seq]

    def prepareMTData(self, sequences, seed=123, do_shuffle=False):
        inputs, outputs = sequences

        decoder_inputs = np.array([sequence[:-1] for sequence in outputs])  # ignore "SENTEND"

        #decoder_outputs = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in outputs ] )
        decoder_outputs = np.array([sequence[1:] for sequence in outputs])  # ignore "SENTSTART"

        matching_input_token = []

        for cur_outputs, cur_inputs in zip(decoder_outputs, inputs):  # (ignore "SENTSTART") vs (original)
            tmp = []
            for output_token in cur_outputs:  # for-each (ignore "SENTSTART")
                idx = np.zeros(len(cur_inputs), dtype=np.float32)
                for j, token in enumerate(cur_inputs):  # for-each (original sequence)
                    if token <= 3:  # ==self.word_to_idx[self.pad_word]:
                        continue
                    if token == output_token:
                        idx[j] = 1.0
                tmp.append(idx)  # tmp: [ 01-vec for 1st-tok, 01-vec for 2nd-tok, ... ]
            matching_input_token.append(tmp)
        matching_input_token = np.array(matching_input_token)
        encoder_inputs = np.array(inputs)

        if do_shuffle:
            #shuffling
            indices = np.arange(encoder_inputs.shape[0])
            np.random.seed(seed)
            np.random.shuffle(indices)
        print ("np.sum(np.sum(np.sum(matching_input_token))) = ", np.sum(np.sum(np.sum(matching_input_token))))
        return encoder_inputs, decoder_inputs, decoder_outputs, matching_input_token
