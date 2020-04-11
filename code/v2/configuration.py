import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_PATH, "data")
JIEBA_DICT_PATH = os.path.join(DATA_PATH, "jieba_dict")
TMP_PATH = os.path.join(DATA_PATH, "tmp_v2")

BABEL_PATH = os.path.join(DATA_PATH, "dl-BABEL-Cantonese", "data")
BABEL_ORI_PATH = os.path.join(BABEL_PATH, "IARPA_BABEL_BP_101")

HKCANCOR_PATH = os.path.join(DATA_PATH, "dl-HKCANCOR")
HKCANCOR_ORI_PATH = os.path.join(HKCANCOR_PATH, "hkcancor-utf8", "utf8")

WIKI_PATH = os.path.join(DATA_PATH, "dl-Wikipedia-YUE")
WIKI_ORI_PATH = os.path.join(WIKI_PATH, "json")

UD_PATH = os.path.join(DATA_PATH, "dl-UD_Cantonese-Chinese")
UD_STDCH_PATH = os.path.join(UD_PATH, "UD_Chinese-HK-master")
UD_STDCH_CONLLU = os.path.join(UD_STDCH_PATH, "zh_hk-ud-test.conllu")
UD_CANTO_PATH = os.path.join(UD_PATH, "UD_Cantonese-HK-master")
UD_CANTO_CONLLU = os.path.join(UD_CANTO_PATH, "yue_hk-ud-test.conllu")
