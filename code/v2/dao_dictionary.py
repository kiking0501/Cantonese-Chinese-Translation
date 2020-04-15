import os
from configuration import (
    CHI_CUHK_DATABASE_PATH, KFCD_PATH, KFCD_DICT_PATH, JIEBA_DICT_PATH
)


def read_chi_cuhk_database():
    '''
        canto to stdch mapping  (one-to-many)
    '''
    stdch2canto, canto2stdch = {}, {}
    with open(CHI_CUHK_DATABASE_PATH) as f:
        lines = f.readlines()
        for l in lines:
            canto_w, stdch_list = l.strip().split(' ')
            for stdch_w in stdch_list.split('/'):
                stdch2canto[stdch_w] = canto_w
            canto2stdch[canto_w] = stdch_list.split('/')
    return canto2stdch, stdch2canto


def read_kfcd(bilingual_only=False):
    if not os.path.exists(KFCD_DICT_PATH):
        raise IOError("You need to download cidian_zhyue-kfcd.zip (漢語詞典) from\n"
                      " http://www.kaifangcidian.com/xiazai/\n"
                      " extract and place the cidian_zhyue-kfcd folder\n"
                      " under %s!" % KFCD_PATH)

    canto2stdch, stdch2canto = {}, {}
    with open(KFCD_DICT_PATH) as f:
        lines = f.readlines()
        for ind, l in enumerate(lines):
            if ind < 9:
                continue

            line = l.strip().split('\t')
            canto_w, stdch_list = line[0], line[2].split('，') if len(line) > 2 else []

            canto2stdch[canto_w] = []
            for stdch in stdch_list:
                ### same filter condition in /code/v1/crawl_dict_map
                if len(stdch) > 4 or (len(stdch) - len(canto_w) > 2):
                    continue
                canto2stdch[canto_w].append(stdch)
                stdch2canto[stdch] = canto_w

    if bilingual_only:
        canto2stdch = {k: v for k, v in canto2stdch.items() if v}
    return canto2stdch, stdch2canto


def create_canto_dictionary(output_file="dict.txt.all_canto"):
    from configuration import PYCANTO_PATH, UD_CANTO_CONLLU
    from dao_hkcancor import read_clean_transcription_combine as read_hkcancor
    from dao_UD import read_UD_sentences
    from dao_babel import read_clean_transcription_combine as read_babel

    pos_dict = {}
    freq_dict = {}

    freq_dict["pycanto"] = {}
    with open(PYCANTO_PATH) as f:
        lines = f.readlines()
        for l in lines:
            line = l.strip().split(' ')
            freq_dict["pycanto"][line[0]] = int(line[1])
            pos_dict[line[0]] = line[2]

    freq_dict["hkcancor"] = {}
    for sen in read_hkcancor():
        for w in sen:
            if w not in freq_dict["hkcancor"]:
                freq_dict["hkcancor"][w] = 1
            else:
                freq_dict["hkcancor"][w] += 1

    freq_dict["UD"] = {}
    for sen in read_UD_sentences(UD_CANTO_CONLLU, tokenize=True):
        for w in sen:
            if w not in freq_dict["UD"]:
                freq_dict["UD"][w] = 1
            else:
                freq_dict["UD"][w] += 1

    freq_dict["babel"] = {}
    for sen in read_babel():
        for w in sen:
            if w not in freq_dict["babel"]:
                freq_dict["babel"][w] = 1
            else:
                freq_dict["babel"][w] += 1

    all_canto_dict = {}

    for dataset, d in freq_dict.items():
        for k, v in d.items():
            if k not in all_canto_dict:
                all_canto_dict[k] = v
            else:
                all_canto_dict[k] += v

    cuhk_dict, _ = read_chi_cuhk_database()
    for k in cuhk_dict:
        if k not in all_canto_dict:
            all_canto_dict[k] = 0
        else:
            all_canto_dict[k] += 1

    kfcd_dict, _ = read_kfcd()
    for k in kfcd_dict:
        if k not in all_canto_dict:
            all_canto_dict[k] = 0
        else:
            all_canto_dict[k] += 1

    with open(os.path.join(JIEBA_DICT_PATH, output_file), "w") as f:
        for k, v in sorted(all_canto_dict.items(), key=lambda x: (-x[1], -len(x[0]), x[0])):
            if not k:
                continue
            if k in pos_dict:
                f.write("%s %d %s\n" % (k, v, pos_dict[k]))
            elif v > 0:
                f.write("%s %d\n" % (k, v))
            else:
                f.write("%s\n" % k)
        print("%s saved." % f.name)


if __name__ == '__main__':
    create_canto_dictionary()
