import os
import sentencepiece as spm
from configuration import DATA_PATH, SPM_PATH, ALL_CANTO_SENTENCES

SAVE_DIR = os.path.join(SPM_PATH, "cantonese")


DATA_SELECT = {
    'wiki': True,
    'babel': True,
    'hkcancor': True,
}

PARAM = {
    'input': os.path.join(DATA_PATH, "static", ALL_CANTO_SENTENCES),
    'model_prefix': os.path.join(SAVE_DIR, '_'.join(['spm'] + [k for k, v in sorted(DATA_SELECT.items()) if v])),
    'vocab_size': 32000,
    'shuffle_input_sentence': "true",
    'model_type': 'bpe',
    'bos_id': -1,
    'eos_id': -1,
}


SPM_COMMAND = ('--input={input}'
               ' --model_prefix={model_prefix}'
               ' --vocab_size={vocab_size}'
               ' --shuffle_input_sentence={shuffle_input_sentence}'
               ' --model_type={model_type}'
               ' --bos_id={bos_id}'
               ' --eos_id={eos_id}').format(**PARAM)

spm.SentencePieceTrainer.Train(SPM_COMMAND)
