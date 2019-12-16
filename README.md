# Cantonese-Chinese-Translation

An experimental study on *Standard-Chinese* to *Cantonese* translator models.

Two major approaches are included:

- Copy-Enriched Seq2Seq Models (Jhamtani., 2017)
- Enriched dictionary table by Translation-Matrix (Mikolov., 2013)

This code is initiated from the work by Jhamtani; link to the original code: https://github.com/harsh19/Shakespearizing-Modern-English


# Requirements
- Python 3.5
- Tensorflow 1.1.0 (framework)

# Instructions to run:
Change working directory to /code/main/

#### Initialization : 
- Run: </br>
`python mt_main.py init` </br>
to download the pre-trained cantonese and chinese embeddings from [fastText](https://fasttext.cc/docs/en/pretrained-vectors.html); and also to build token-dictionaries for the chinese tokenizer

#### Preprocessing: 
- First run initialization
- Run: </br>
`python mt_main.py preprocessing` </br>
to preprocess and save train/valid/test data.
- The used dictionaries for tokenization can be changed at `prepro.py`

#### Copy-Enriched Seq-to-Seq Model: 
- First run pre-processing
- Run
`python mt_main.py train <iter_num> <output_model_name>` </br> or
`python mt_main.py <valid/test> <saved_model_name> <inference_type>`
- Training settings can be modified at `configuration.py`
- Trained models and results are saved in /data/tmp/
- Link to the original paper: https://arxiv.org/abs/1707.01161

#### Translation-Matrix Model: 
- First run pre-processing
- [Download](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh_yue.zip) and get the cantonese language model (wiki.zh_yue.bin), put under data/embedding/
- Run
`python mt_translation_matrix.py` </br>
  and a linear projection matrix between the embedding spaces would be learnt using Stochoastic Gradient Descent
- Trained matrix, interal validation accuracy and finaly results are saved in /data/translation_matrix /
- Link to the original paper: https://arxiv.org/pdf/1309.4168.pdf

### Evaluation:
- First run pre-processing
- Run
`python baseline_as_it_is.py` or
`python baseline_dictionary.py` 
to check performance of the two baseline methods
- Results are saved in /data/baselines

# Details of Implementation

### Data Collection
- Movie transcripts are used to create a collection of sentences in pairs {Standard-Chinese(繁), Cantonese (粵)} as the parallel corpora. (/data/transcript/)
- A Cantonese-SC dictionary mapping consisting ~1600 entries are created from the online database from [A Comparative Study of Modern Chinese and Cantonese in the Development of Teaching Resource](https://apps.itsc.cuhk.edu.hk/hanyu/Page/Intro.aspx) (/data/static/canto2stdch_full.dict, see script at `crawl_dict_map.py`)


### Preprocessing
- Tokenziation by the tool [Jieba]( https://github.com/fxsjy/jieba) with customized dictionaries. Cantonese sentences are tokenized based on token words from [PyCantonese]( https://github.com/jacksonllee/pycantonese) and also available words in the embedding.
- Pre-trained embeddings for Chinese and Cantonese are downloaded from [FastText](https://fasttext.cc/docs/en/pretrained-vectors.html) (Wiki word vectors)
- Convertion of traditional-vs-simplified chinese characters is done by the python wrapper [OpenCC-Python](https://github.com/yichen0831/opencc-python) with respect to [Open Chinese Convert](https://github.com/BYVoid/OpenCC). 

### Evaluation
- BLEU metric evaluation is provided by the tookit [MOSES](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)
- BLEU evaluation is adjusted to consider only single-Chinese-character tokenization


# Troubleshoot

### BLEU evaluation
- *OSError: [Errno 12] Cannot allocate memory*: Make sure one have enough RAM; restart the computer, or try [adding a new swapfile](https://askubuntu.com/questions/927854/how-do-i-increase-the-size-of-swapfile-without-removing-it-in-the-terminal) (e.g. make to 4G total).


# Reference 
(To be updated)
```
@article{jhamtani2017shakespearizing,
  title={Shakespearizing Modern Language Using Copy-Enriched Sequence-to-Sequence Models},
  author={Jhamtani, Harsh and Gangal, Varun and Hovy, Eduard and Nyberg, Eric},
  journal={EMNLP 2017},
  volume={6},
  pages={10},
  year={2017}
}

@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```