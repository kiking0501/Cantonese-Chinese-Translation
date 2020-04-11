import pyconll
from configuration import (
    UD_PATH,
    UD_STDCH_PATH, UD_CANTO_PATH,
    UD_STDCH_CONLLU, UD_CANTO_CONLLU
)
import os

if not os.path.exists(UD_CANTO_CONLLU):
    raise IOError("You need to download yue_hk-ud-test.conllu from\n"
                  " https://github.com/UniversalDependencies/UD_Cantonese-HK\n"
                  " and place it under %s!\n" % UD_CANTO_PATH)
if not os.path.exists(UD_STDCH_CONLLU):
    raise IOError("You need to download zh_hk-ud-test.conllu from\n"
                  " https://github.com/UniversalDependencies/UD_Chinese-HK\n"
                  " and place it under %s!\n" % UD_STDCH_PATH)


def read_UD(file_path):
    return pyconll.load_from_file(file_path)


def read_UD_sentences(file_path):
    UD_file = read_UD(file_path)
    sentences = [[token.form for token in sentence]
                 for sentence in UD_file]
    return sentences


def save_UD_sentences_for_TER(tokenize=False, stdch_file="UD_stdch.txt", canto_file="UD_canto.txt"):
    stdch_sen = read_UD_sentences(UD_STDCH_CONLLU)
    canto_sen = read_UD_sentences(UD_CANTO_CONLLU)
    with open(os.path.join(UD_PATH, stdch_file), "w") as stdch_f:
        with open(os.path.join(UD_PATH, canto_file), "w") as canto_f:
            for stdch, canto in zip(stdch_sen, canto_sen):
                if tokenize:
                    stdch_f.write(' '.join(stdch) + ' ' + '(%s)' % ''.join(stdch) + '\n')
                    canto_f.write(' '.join(canto) + ' ' + '(%s)' % ''.join(stdch) + '\n')
                else:
                    stdch_f.write(''.join(stdch) + ' ' + '(%s)' % ''.join(stdch) + '\n')
                    canto_f.write(''.join(canto) + ' ' + '(%s)' % ''.join(stdch) + '\n')
        print("%s | %s saved. " % (stdch_file, canto_file))


if __name__ == '__main__':
    save_UD_sentences_for_TER()
    '''
      Example command line for TER evaluation
      $ cd ./code/v2
      $ java -jar tercom-0.7.25/tercom.7.25.jar -r ../../data/dl-UD_Cantonese-Chinese/UD_canto.txt -h ../../data/dl-UD_Cantonese-Chinese/UD_stdch.txt -n ../../data/tmp_v2/TER.r_canto_h_stdch
    '''
