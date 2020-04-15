import os


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_PATH, "data")
JIEBA_DICT_PATH = os.path.join(DATA_PATH, "jieba_dict")
EMB_PATH = os.path.join(DATA_PATH, "embedding")
SPM_PATH = os.path.join(DATA_PATH, "spm_tokenizer")
EVAL_PATH = os.path.join(BASE_PATH, "code", "eval")


### Generated / Trained ###
##############################################################

# under JIEBA_DICT_PATH
JIEBA_CANTO = "dict.txt.all_canto"  # 57,019 tokens

# under EMB_PATH
EMBEDDING_CANTO = "jieba-fasttext_babel_hkcancor_wiki"  # 68,652 tokens

# under STATIC_PATH
ALL_CANTO_SENTENCES = "all_canto_sentences.txt"  # 527,442 sentences
BI_SENTENCES = {k: "bilingual_sentences_%s.txt" % k for k in ["dev", "train", "valid", "test"]}  # Train: 7657 / Valid: 851 / Test: 955

# under SPM_PATH
SPM_CANTO = "spm_babel_hkcancor_wiki"  # 32,000 tokens


### Cantonese Dictionaries ###
##############################################################

### Cantonese Linguistics and NLP in Python
### http://pycantonese.org/
PYCANTO_PATH = os.path.join(JIEBA_DICT_PATH, "dict.txt.pycanto")

### A Comparative Study of Modern Chinese and Canotnese in the Development of Teaching Resources", CUHK, 2001  (現代標準漢語與粵語對照資料庫)
### https://apps.itsc.cuhk.edu.hk/hanyu/
### 1573 Cantonese Words -> 2573 Mandarin Words
CHI_CUHK_DATABASE_PATH = os.path.join(DATA_PATH, "static", "canto2stdch_full.dict")

### Kaifangcidian 開放粵語詞典
### http://www.kaifangcidian.com/han/yue/
### 40041 Cantonese Words -> 16713 Mandarin Words
KFCD_PATH = os.path.join(DATA_PATH, "dl-kfcd", "cidian_zhyue-kfcd")
KFCD_DICT_PATH = os.path.join(KFCD_PATH, "cidian_zhyue-ft-kfcd-ylshu-2019623.txt")


### Language Datasets ###
##############################################################

### Manually annotated parallel corpus (8,198 sentence pairs)
### Movie transcripts from selected Stephen-Chow's movie collections
MOVIE_PATH = os.path.join(DATA_PATH, "MOVIE-transcript")


### The Cantonese-Mandarin Parallel Dependency Treebank
### https://github.com/UniversalDependencies/UD_Cantonese-HK
### https://github.com/UniversalDependencies/UD_Chinese-HK
UD_PATH = os.path.join(DATA_PATH, "dl-UD_Cantonese-Chinese")
UD_STDCH_PATH = os.path.join(UD_PATH, "UD_Chinese-HK-master")
UD_STDCH_CONLLU = os.path.join(UD_STDCH_PATH, "zh_hk-ud-test.conllu")
UD_CANTO_PATH = os.path.join(UD_PATH, "UD_Cantonese-HK-master")
UD_CANTO_CONLLU = os.path.join(UD_CANTO_PATH, "yue_hk-ud-test.conllu")

### The Tatoeba Project
### http://www.manythings.org/anki/
TATOEBA_PATH = os.path.join(DATA_PATH, "dl-Tatoeba")
TATOEBA_STDCH_PATH = os.path.join(TATOEBA_PATH, "cmn-eng", "cmn.txt")
TATOEBA_CANTO_PATH = os.path.join(TATOEBA_PATH, "yue-eng", "yue.txt")

### Wikimedia Downloads, database backup dumps, zh_yue collection
### https://dumps.wikimedia.org/zh_yuewiki/
WIKI_PATH = os.path.join(DATA_PATH, "dl-Wikipedia-YUE", "20200401")
WIKI_ORI_PATH = os.path.join(WIKI_PATH, "json")

### The Hong Kong Cantonese Corpus (HKCanCor) 香港粵語語料庫
### https://github.com/fcbond/hkcancor/
HKCANCOR_PATH = os.path.join(DATA_PATH, "dl-HKCANCOR")
HKCANCOR_ORI_PATH = os.path.join(HKCANCOR_PATH, "hkcancor-utf8", "utf8")

### IARPA Babel Cantonese Language Pack IARPA-babel101b-v0.4c
### https://catalog.ldc.upenn.edu/LDC2016S02
BABEL_PATH = os.path.join(DATA_PATH, "dl-BABEL-Cantonese", "data")
BABEL_ORI_PATH = os.path.join(BABEL_PATH, "IARPA_BABEL_BP_101")


SEN_INFO = {
    "Movie collections": {
        "Corpus type": "parallel",
        "Tokenized": False,
        "POS tagged": False,
        "Romanized": False,
        "Register": "low",
        "No. Sentences": 8198,
    },
    "UD-Treebank": {
        "Corpus type": "parallel",
        "Tokenized": True,
        "POS tagged": True,
        "Romanized": False,
        "Register": "low,high",
        "No. Sentences": 1004,
    },
    "Tatoeba": {
        "Corpus type": "parallel",
        "Tokenized": False,
        "POS tagged": False,
        "Romanized": False,
        "Register": "low",
        "No. Sentences": 437,
    },
    "Wikipedia": {
        "Corpus type": "monolingual",
        "Tokenized": False,
        "POS tagged": False,
        "Romanized": False,
        "Register": "high",
        "No. Sentences": 427567,   # (20200401: 80021/427567) VS (20170720: 55645/306466)
    },
    "HKCanCor": {
        "Corpus type": "monolingual",
        "Tokenized": True,
        "POS tagged": True,
        "Romanized": True,
        "Register": "low",
        "No. Sentences": 9048,
    },
    "Babel": {
        "Corpus type": "monolingual",
        "Tokenized": True,
        "POS tagged": False,
        "Romanized": True,
        "Register": "low",
        "No. Sentences": 90827,
    }
}

TOK_INFO = {
    "PyCanto": {
        "No. Tokens": 7085,
        "POS tagged": True,
        "Translated": False,
    },
    "HKCanCor": {
        "No. Tokens": 378,  # wPyCanto: 370
        "POS tagged": True,
        "Translated": False,
    },
    "UD-Treebank": {
        "No. Tokens": 1527,  # wPyCanto: 943, wBabel: 993
        "POS tagged": True,
        "Translated": False,
    },
    "Babel": {
        "No. Tokens": 20277,  # wPyCanto: 3562
        "POS tagged": False,
        "Translated": False,
    },
    "Chi CUHK Database": {
        "No. Tokens": 1573,  # wPyCanto: 694, wBabel: 1037
        "POS tagged": False,
        "Translated": False,
    },
    "KFCD": {
        "No. Tokens": 40041,  # wPyCanto: 3372, wBabel: 6212
        "POS tagged": False,
        "Translated": True,
    }

}

if __name__ == '__main__':
    import pandas as pd

    sen_info = pd.DataFrame.from_dict(SEN_INFO)[
        ["Movie collections", "UD-Treebank", "Tatoeba", "Wikipedia", "HKCanCor", "Babel"]
    ]
    sen_info = sen_info.reindex(
        ["Corpus type", "Tokenized", "POS tagged", "Romanized", "Register", "No. Sentences"]
    )

    tok_info = pd.DataFrame.from_dict(TOK_INFO)[
        ["PyCanto", "HKCancor", "UD-Treebank", "Babel", "Chi CUHK Database", "KFCD"]
    ]
    tok_info = tok_info.reindex(
        ["No. Tokens", "Translated"]
    )
