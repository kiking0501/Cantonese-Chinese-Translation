import configuration as config
import jieba
import dao
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np


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

    OUT_TOK = "dict.txt.pycanto-custom_canto_wiki"
    INP_TOK = "dict.txt.big_trad"
    # OUT_TOK = "char"
    # INP_TOK = "char"

    def __init__(self):
        self.unknown_word = "UNK".lower()
        self.sent_start = "SENTSTART".lower()
        self.sent_end = "SENTEND".lower()
        self.pad_word = "PADWORD".lower()
        self.special_tokens = [self.sent_start, self.sent_end, self.pad_word, self.unknown_word]
        self.word_counters, self.word_to_idx, self.word_to_idx_ctr, self.idx_to_word = self.initVocabItems()

    def initVocabItems(self):
        word_counters = {}
        word_to_idx = {}
        word_to_idx_ctr = 0
        idx_to_word = {}

        word_to_idx[self.pad_word] = word_to_idx_ctr  # 0 is for padword
        idx_to_word[word_to_idx_ctr] = self.pad_word
        word_counters[self.pad_word] = 1
        word_to_idx_ctr += 1

        word_to_idx[self.sent_start] = word_to_idx_ctr  # 1 is for SENTSTART
        word_counters[self.sent_start] = 1
        idx_to_word[word_to_idx_ctr] = self.sent_start
        word_to_idx_ctr += 1

        word_to_idx[self.sent_end] = word_to_idx_ctr  # 2 is for SENTEND
        word_counters[self.sent_end] = 1
        idx_to_word[word_to_idx_ctr] = self.sent_end
        word_to_idx_ctr += 1

        word_to_idx[self.unknown_word] = word_to_idx_ctr  # 3 is for UNK
        word_counters[self.unknown_word] = 1
        idx_to_word[word_to_idx_ctr] = self.unknown_word
        word_to_idx_ctr += 1

        return word_counters, word_to_idx, word_to_idx_ctr, idx_to_word

    def preprocess(self, filename, dict_txt):
        ''' Modification:
                Tokenization with Jieba + Customized Dictionary
        '''
        return dao.get_sen2tok(filename, dict_txt)

    def loadVocab(self, split):
        print("="*5, " loadData: split = ", split)

        inputs = self.preprocess(self.INP_SRC[split], self.INP_TOK)
        outputs = self.preprocess(self.OUT_SRC[split], self.OUT_TOK)

        word_to_idx = self.word_to_idx
        idx_to_word = self.idx_to_word
        word_to_idx_ctr = self.word_to_idx_ctr
        word_counters = self.word_counters

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

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = len(word_to_idx)
        self.word_to_idx_ctr = word_to_idx_ctr
        self.word_counters = word_counters

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

    def _generate_sequences(self, sen_toks):
        word_to_idx = self.word_to_idx
        idx_to_word = self.idx_to_word
        word_to_idx_ctr = self.word_to_idx_ctr

        # generate sequences
        sequences = []
        texts = sen_toks
        for text in texts:
            tmp = [word_to_idx[self.sent_start]]
            for token in text:
                if token not in word_to_idx:
                    tmp.append(word_to_idx[self.unknown_word])
                else:
                    tmp.append(word_to_idx[token])
            tmp.append(word_to_idx[self.sent_end])
            sequences.append(tmp)
        return sequences

    def generate_inp_sequences(self, inp_sen_toks):
        sequences_input = self._generate_sequences(inp_sen_toks)
        sequences_input = pad_sequences(
            sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
        return sequences_input

    def generate_out_sequences(self, out_sen_toks):
        sequences_output = self._generate_sequences(out_sen_toks)
        sequences_output = pad_sequences(
            sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')
        return sequences_output

    def loadData(self, split):
        print("="*5, " loadData: split = ", split)
        inputs = self.preprocess(self.INP_SRC[split], self.INP_TOK)
        outputs = self.preprocess(self.OUT_SRC[split], self.OUT_TOK)
        sequences_input = self.generate_inp_sequences(inputs)
        sequences_output = self.generate_out_sequences(outputs)
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


def modifyParamsWithPrepro(params, preprocessing, verbose=True):
        ''' Add preprocessing details to params
        '''
        params['vocab_size'] = preprocessing.vocab_size
        params['preprocessing'] = preprocessing

        #Pretrained embeddibngs
        if params['use_pretrained_embeddings']:
            pretrained_embeddings = {
                'canto': pickle.load(open(params['canto_embedding_path'], "rb")),
                'stdch': pickle.load(open(params['stdch_embedding_path'], "rb")),
            }
            word_to_idx = preprocessing.word_to_idx
            embedding_matrix = {
                'canto': np.random.rand(params['vocab_size'], params['embeddings_dim']),
                'stdch': np.random.rand(params['vocab_size'], params['embeddings_dim'])
            }
            not_found_count = {'stdch': 0, 'canto': 0}
            for src, matrix_name in [('stdch', 'encoder_embeddings_matrix'),
                                     ('canto', 'decoder_embeddings_matrix')]:
                not_found_count = 0
                for token, idx in word_to_idx.items():
                    if token in pretrained_embeddings[src]:
                        embedding_matrix[src][idx] = pretrained_embeddings[src][token]
                    else:
                        not_found_count += 1
                        if not_found_count < 10 and verbose:
                            print ("No pretrained embedding for (only first 10 such cases will be printed. other prints are suppressed) ",token)
                if verbose:
                    print ("(%s)not found count = " % src, not_found_count)
                params[matrix_name] = embedding_matrix[src]

        if params['use_additional_info_from_pretrained_embeddings']:
            for src, matrix_name in [('stdch', 'encoder_embeddings_matrix'),
                                     ('canto', 'decoder_embeddings_matrix')]:
                additional_count = 0
                tmp = []
                for token in pretrained_embeddings[src]:
                    if token not in preprocessing.word_to_idx:
                        preprocessing.word_to_idx[token] = preprocessing.word_to_idx_ctr
                        preprocessing.idx_to_word[preprocessing.word_to_idx_ctr] = token
                        preprocessing.word_to_idx_ctr += 1
                        tmp.append(pretrained_embeddings[src][token])
                        additional_count += 1
                if verbose:
                    print ("additional_count(%s) = %s " % (src, additional_count))
                tmp = np.array(tmp)
                params[matrix_name] = np.vstack([params[matrix_name], tmp])
            #print "New vocab size = ",params['vocab_size']
            params['vocab_size'] = preprocessing.word_to_idx_ctr

        return params
