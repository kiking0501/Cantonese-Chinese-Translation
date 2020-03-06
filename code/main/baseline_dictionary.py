from collections import defaultdict
import re
import configuration as config
import os
from dao import _replace_space, char_cut

## Output Settings
DICT_OUTPUT_FILE = "dictionary_baseline.output"
BLEU_OUTPUT_FILE = "dictionary_baseline.BLEU"

### Validation Settings
# DICT_FILE = "static/canto2stdch_full.dict"
# TEST_INPUT_TOKENIZED = "valid.stdch.sent.tok.dict.txt.big_trad"
# TEST_GT_CHAR = "valid.canto.sent.tok.char"

## Test Settings
DICT_FILE = "static/simple.dict"
TEST_INPUT_TOKENIZED = "test.stdch.sent.tok.dict.txt.big_trad"
TEST_GT_CHAR = "test.canto.sent.tok.char"


def readDict(dictFile):
    with open(dictFile) as f:
        lines = f.readlines()
    #lines = [x.lower() for x in lines]
    lines = [x.split() for x in lines]
    dictionary = {}
    for (w1, w2_list) in lines:
        for w2 in w2_list.split("/"):
            dictionary[w2] = w1
    return dictionary


def create_regex_map(dictionary):
    regex_map = defaultdict(list)
    for w in dictionary:
        regex_map[len(w)].append(w)

    regex_map = {
        # mapping words except included in "{}"
        k: re.compile("(%s)(?![^{]*})" % "|".join(regex_map[k]))
        for k in regex_map.keys()
    }
    return regex_map


def map_sentence_by_regex(sentence, dictionary, regex_map):
    def dictrepl(matchobj):
        return '{' + dictionary[matchobj.group()] + '}'

    new_sentence = sentence
    for k in sorted(regex_map.keys(), reverse=True):
        new_sentence = regex_map[k].sub(dictrepl, new_sentence)
    return new_sentence


def generate_output(dictFile, srcFile, trgFile, map_type="replace"):
    ''' map_type can either be "replace" or "regex"
            "replace": tokenize before mapping
            "regex" : direct mapping
    '''
    dictionary = readDict(dictFile)
    regex_map = create_regex_map(dictionary)

    with open(srcFile) as src_f:
        srcLines = src_f.readlines()

        with open(trgFile, "w") as trg_f:
            for sentence in srcLines:
                new_sentence = sentence
                if map_type == 'regex':
                    new_sentok = []
                    for st in sentence[:-1].split(" "):
                        new_sentok += [st, " "]
                    new_sentok = new_sentok[:-1]
                    new_sentok = _replace_space(new_sentok)
                    new_sentence = " ".join(new_sentok)
                    new_sentence = map_sentence_by_regex(new_sentence, dictionary, regex_map)
                    new_sentence = re.sub('{|}', '', new_sentence)
                elif map_type == "replace":
                    new_sentok = sentence[:-1].split(" ")
                    new_sentok = [dictionary.get(st, st) for st in new_sentok]
                    new_sentok = char_cut(new_sentok)
                    new_sentence = " ".join(new_sentok)
                else:
                    raise ValueError("Unknown map_type %s!" % map_type)
                trg_f.write("%s\n" % new_sentence)
        print("%s saved." % trg_f.name)

if __name__ == '__main__':
    baseline_dir = os.path.join(config.data_dir, "baselines")
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    dictFile = os.path.join(config.data_dir, DICT_FILE)
    testInput = os.path.join(config.data_dir, TEST_INPUT_TOKENIZED)
    testGT = os.path.join(config.data_dir, TEST_GT_CHAR)
    dictOutputFile = os.path.join(baseline_dir, DICT_OUTPUT_FILE)

    generate_output(dictFile, testInput, dictOutputFile)

    print("perl multi-bleu.perl -lc " + testGT + " < " + dictOutputFile)
    BLEUOutput = os.popen("perl multi-bleu.perl -lc " + testGT + " < " + dictOutputFile).read()
    with open(os.path.join(baseline_dir, BLEU_OUTPUT_FILE), "w") as f:
        print(BLEUOutput)
        f.write(BLEUOutput)
