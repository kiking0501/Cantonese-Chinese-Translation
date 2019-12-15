import os
from opencc import OpenCC
import srt
import numpy as np
import pycantonese as pc
from collections import defaultdict
import jieba
import pickle
from determine_lang import DetermineLanguage
from configuration import data_dir
import requests
import shutil


s2t = OpenCC('s2t')
t2c = OpenCC('t2s')

DL = DetermineLanguage()


def translate_jieba_dict():
    print("Translating Jieba Dict to Traditional Chinese...")
    with open(os.path.join(data_dir, "jieba_dict", "dict.txt.big")) as f:
        cnt = 0
        trad_f = open(os.path.join(data_dir, "jieba_dict", "dict.txt.big_trad"), "w")
        line = f.readline()
        repeat_set = set()
        while line:
            trad = s2t.convert(line)
            if trad in repeat_set:
                line = f.readline()
                continue
            repeat_set.add(trad)
            trad_f.write(trad)
            # print("%s -> %s" % (line.strip('\n'), trad.strip('\n')))
            line = f.readline()
            cnt += 1

        print("Total: %d" % cnt)
        print("%s saved." % trad_f.name)


def create_jieba_dict_pycanto():
    print("Creating Cantonese Jieba Dict...")
    corpus = pc.hkcancor()
    freq = corpus.word_frequency()

    pycanto_dict = defaultdict(int)
    for l in sorted(corpus.tagged_words(), key=lambda x: (x[0], len(x[0]))):
        pycanto_dict[l] += 1

    # print("Tagged Words / Freq / Repeated")
    with open(os.path.join(data_dir, "jieba_dict", "dict.txt.pycanto"), "w") as f:
        for (ww, c) in sorted(pycanto_dict.items(), key=lambda x: (-x[1], x[0][0], len(x[0][0]))):
            # print(ww, ':', freq[ww[0]], '/', c)
            f.write("%s %s %s\n" %(ww[0], freq[ww[0]],  ww[1].lower()))

        print("Total: %d" % (len(pycanto_dict)))
        print("%s saved." % f.name)


T_NUM = {
    '01.srt': 650
}


def read_transcript(file_name, data_num=None):
    if data_num is None:
        data_num = T_NUM.get(file_name)
    if data_num is None:
        raise ValueError("%s unknown; Please specify the number of lines by data_num!" % file_name)

    with open(os.path.join(data_dir, "transcript", file_name), 'r') as f:
        data = f.read()

    li = [s.content.partition('\n')[0:3:2]
          for ind, s in enumerate(srt.parse(data))
          if ind < data_num]
    return li


def train_valid_split(data, train_ratio=0.8):
    ''' data should be a list of tuples of sentences (std-chinese, cantonese) '''
    np.random.seed(2046)
    shuffle_ids = list(range(len(data)))
    np.random.shuffle(list(range(len(data))))
    num_train = int(len(data) * train_ratio)
    train_ids, valid_ids = shuffle_ids[:num_train], shuffle_ids[num_train:]
    return [data[id_] for id_ in train_ids], [data[id_] for id_ in valid_ids]


def save_train_valid(train_data, valid_data):
    ''' train_data / valid_data should be a list of tuples of sentences (std-chinese, cantonese) '''

    for split, data in [("train", train_data), ("valid", valid_data)]:
        stdch_f = open(os.path.join(data_dir, "%s.stdch.sent" % split), "w")
        canto_f = open(os.path.join(data_dir, "%s.canto.sent" % split), "w")
        for li in data:
            stdch_f.write("%s\n" % li[0])
            canto_f.write("%s\n" % li[1])
        stdch_f.close()
        canto_f.close()
        print("%s saved." % stdch_f.name)
        print("%s saved." % canto_f.name)
        save_sen2tok("%s.stdch.sent" % split, 'char')
        save_sen2tok("%s.canto.sent" % split, 'char')


def load_jieba(dict_txt=None):
    ''' load jieba with the stated dictionary txt (valid after ver 0.28 '''
    if dict_txt is None:
        dict_txt = "dict.txt.big"
    jieba.set_dictionary(os.path.join(data_dir, "jieba_dict", dict_txt))
    print("Successfully set Jieba-dict to '%s'. " % dict_txt)


def char_cut(ori_sentok):
    ''' cut all chinese tokens into single characters '''
    sentok = []
    for ori_st in ori_sentok:
        if DL.lang(ori_st) == "CHINESE":
            sentok += list(ori_st)
        else:
            sentok.append(ori_st)
    return sentok


def _replace_space(ori_sentok):
    ''' space is reserved for tokenization
        so replace original space by comma
    '''
    new_sentok = []
    for ind, st in enumerate(ori_sentok[1:-1]):
        if not st.strip():
            left = bool(DL.lang(ori_sentok[ind]) == "ENGLISH")
            right = bool(DL.lang(ori_sentok[ind+2]) == "ENGLISH")
            if not (left and right):
                new_sentok.append(",")
                continue
        new_sentok.append(st)
    return [ori_sentok[0]] + new_sentok + [ori_sentok[-1]]


def save_sen2tok(file_name, dict_txt):

    print("Tokenizing %s with %s ..." % (file_name, dict_txt))
    with open(os.path.join(data_dir, file_name), "r") as f:
        sen_list = f.readlines()
    # num = int(sen_list[0])
    # sen_list = sen_list[1:]

    if dict_txt == 'char':
        load_jieba()
        sentok_list = [list(jieba.cut(sen[:-1], cut_all=False)) for sen in sen_list]  # avoid "\n"
        sentok_list = [_replace_space(sentok) for sentok in sentok_list]
        sentok_list = [char_cut(sentok) for sentok in sentok_list]
        sentok_list = [" ".join(sentok) for sentok in sentok_list]
    else:
        load_jieba(dict_txt)
        sentok_list = [list(jieba.cut(sen[:-1], cut_all=False)) for sen in sen_list]  # avoid "\n"
        sentok_list = [_replace_space(sentok) for sentok in sentok_list]
        sentok_list = [" ".join(sentok) for sentok in sentok_list]

    with open(os.path.join(data_dir, file_name + ".tok." + dict_txt), "w") as f:
        # f.write("%s\n" % num)
        for sentok in sentok_list:
            f.write(sentok + "\n")
            #print(sentok)

        print("Total: %d" % (len(sentok_list)))
        print("%s saved." % f.name)


def download_embedding():
    import requests
    urls = {
        'cantonese': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh_yue.vec',
        'standard_chinese': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec',
    }
    for k, url in urls.items():
        path = os.path.join(data_dir, "embedding", k, url.rpartition('/')[2])
        if not os.path.exists(path):
            print("Downloading %s embedding from %s..." % (k, url))
        with requests.get(url, stream=True) as r:
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            print("%s saved." % path)


def read_embedding(file_name, verbose=False):
    if verbose: print("Reading %s..." % file_name)
    vec_dict = {}
    cnt = 0
    with open(os.path.join(data_dir, "embedding", file_name)) as f:
        line = f.readline()
        total, dim = [int(seg) for seg in line.split(" ")]

        line = f.readline()
        while line:
            l_list = line.split(" ")
            vec_dict[l_list[0]] = [float(seg) for seg in l_list[1:] if seg != '\n']
            #if verbose: print("%d: %s" % (cnt, l_list[0]))
            cnt += 1
            line = f.readline()
        if verbose: print("Count(Total): %d (%d)" % (cnt, total))
    return vec_dict


def save_embedding(file_name, output_name, ensure_trad=True):
    vec_dict = read_embedding(file_name, verbose=True)
    if ensure_trad: vec_dict = {s2t.convert(k): v for k, v in vec_dict.items()}
    output_path = os.path.join(data_dir, "embedding", output_name)
    pickle.dump(vec_dict, open(output_path, "wb"))
    print("%s saved." % output_path)


def merge_dict_txt_with_embedding_tokens(dict_txt, emb_file):
    print("Merging %s with %s..." % (dict_txt, emb_file))
    with open(os.path.join(config.data_dir, "jieba_dict", dict_txt)) as f:
        dt_toks = [l[:-1] for l in f.readlines()]
        tok_set = set([dt.split(' ')[0] for dt in dt_toks])
    emb_toks = pickle.load(
        open(os.path.join(config.data_dir, 'embedding', emb_file), "rb")).keys()
    dt_toks += [et for et in emb_toks if et not in tok_set and dao.DL.lang(et) == 'CHINESE']
    with open(os.path.join(config.data_dir, "jieba_dict", dict_txt + '-' + emb_file.partition('.')[0]), "w") as f:
        for dt in dt_toks:
            f.write("%s 1\n" % dt)
        print("%s saved." % f.name)


def merge_dict_txt_list(dict_txt_list):
    print("Merging %s ..." % (dict_txt_list))
    total_dt_toks = []
    total_tok_set = set()
    for dict_txt in dict_txt_list:
        with open(os.path.join(config.data_dir, "jieba_dict", dict_txt)) as f:
            dt_toks = [l[:-1] for l in f.readlines()]
            for dt in dt_toks:
                tok = dt.split(' ')[0]
                if tok not in total_tok_set:
                    total_tok_set.add(tok)
                    total_dt_toks.append(dt)
    f_name = '-'.join([dict_txt.rpartition('.')[2] for dict_txt in dict_txt_list])
    with open(os.path.join(config.data_dir, "jieba_dict", "dict.txt.%s" % f_name), "w") as f:
        for dt in total_dt_toks:
            f.write("%s\n" % dt)
        print("%s saved." % f.name)


if __name__ == '__main__':
    translate_jieba_dict()
    create_jieba_dict_pycanto()

    data = read_transcript("01.srt")
    train_data, valid_data = train_valid_split(data)
    save_train_valid(train_data, valid_data)

    save_embedding("cantonese/wiki.zh_yue.vec", "canto_wiki.pkl")
    save_embedding("standard_chinese/wiki.zh.vec", "stdch_wiki.pkl")

    merge_dict_txt_with_embedding_tokens(
        "dict.txt.pycanto", "canto_wiki.pkl")

    # merge_dict_txt_list(
    #     ["dict.txt.pycanto", "dict.txt.big_trad"])
