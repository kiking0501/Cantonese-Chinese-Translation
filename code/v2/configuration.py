import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_PATH, "data")

BABEL_PATH = os.path.join(DATA_PATH, "BABEL-Cantonese", "data")
BABEL_ORI_PATH = os.path.join(BABEL_PATH, "IARPA_BABEL_BP_101")

HKCANCOR_PATH = os.path.join(DATA_PATH, "HKCANCOR")
HKCANCOR_ORI_PATH = os.path.join(HKCANCOR_PATH, "hkcancor-utf8", "utf8")

WIKI_PATH = os.path.join(DATA_PATH, "Wikipedia-YUE")
WIKI_ORI_PATH = os.path.join(WIKI_PATH, "Wikipedia-YUE", "json")
