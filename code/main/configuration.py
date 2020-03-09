import os
from cache import CACHE as _CACHE

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_PATH, "data")
PKL_PATH = os.path.join(DATA_PATH, "pkl")


def CACHE(name):
    return _CACHE(name, dir_path=PKL_PATH)


data_dir = DATA_PATH

# GPU
use_gpu = True

# preprocessing params
max_input_seq_length = 25
max_output_seq_length = 25
# #dropout_val = 0.2
do_vocab_pruning = True
max_vocab_size = 12000
transcript_files = ['01', '01-2', '02', '03', '04', '05', '06', '07']

# Pointer or seq2seq
use_pointer = True

# model config
lstm_cell_size = 300
embeddings_dim = 300

use_reverse_encoder = True
share_encoder_decoder_embeddings = True
use_pretrained_embeddings = True
pretrained_embeddings_are_trainable = False
canto_embedding_path = data_dir + "/embedding/canto_wiki.pkl"
stdch_embedding_path = data_dir + "/embedding/stdch_wiki.pkl"
use_additional_info_from_pretrained_embeddings = True  # if some word is not in training data set but is there in pretrained embeddings: mark True to add such words also. Otherwise mark False
use_additional_info_from_pretrained_embeddings = False

# Specific to pointer model
use_sentinel_loss = False
lambd = 2.0
use_context_for_out = True

# general training params
display_step = 1
sample_step = 1
save_step = 1
batch_size = 32


# params
params = {}
params['embeddings_dim'] = embeddings_dim
params['lstm_cell_size'] = lstm_cell_size
params['max_input_seq_length'] = max_input_seq_length
params['max_output_seq_length'] = max_output_seq_length-1  # inputs are all but last element, outputs are al but first element
params['batch_size'] = batch_size
params['use_pretrained_embeddings'] = use_pretrained_embeddings
params['share_encoder_decoder_embeddings'] = share_encoder_decoder_embeddings
params['use_pointer'] = use_pointer
params['canto_embedding_path'] = canto_embedding_path
params['stdch_embedding_path'] = stdch_embedding_path
params['pretrained_embeddings_are_trainable'] = pretrained_embeddings_are_trainable
params['use_additional_info_from_pretrained_embeddings'] = use_additional_info_from_pretrained_embeddings
params['max_vocab_size'] = max_vocab_size
params['do_vocab_pruning'] = do_vocab_pruning
params['use_reverse_encoder'] = use_reverse_encoder
params['use_sentinel_loss'] = use_sentinel_loss
params['lambd'] = lambd
params['use_context_for_out'] = use_context_for_out
