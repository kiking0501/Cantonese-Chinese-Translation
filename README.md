# Cantonese-Chinese-Translation

An experimental *Standard-Chinese* to *Cantonese* translator model using Copy-Enriched Seq2Seq Models.

The code is built on "Jhamtani H., Gangal V., Hovy E. and Nyberg E. Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models" (Workshop on Stylistic Variation, EMNLP 2017).
Link to the original code: https://github.com/harsh19/Shakespearizing-Modern-English

# Requirements
- Python 3.5
- Tensorflow 1.1.0 (framework)

# Instructions to run:

#### Initialization: 
- Change working directory to code/main/
- Run: </br>
`python mt_main.py init` </br>

#### Preprocessing: 
- First run initialization
- Change working directory to code/main/
- Run: </br>
`python mt_main.py preprocessing` </br>

#### Pointer model: 
- First run pre-processing
- Change working directory to code/main/
- `python mt_main.py train 10 pointer_model` </br>

# Details

### Data Collection
- Movie transcripts are used to create a collection of sentences in pairs {Standard-Chinese(繁), Cantonese (粵)}.

### Preprocessing
- Tokenziation by [Jieba]( https://github.com/fxsjy/jieba). Cantonese sentences are tokenized based on token words from [PyCantonese]( https://github.com/jacksonllee/pycantonese).
- *Raw* embeddings for Chinese and Cantonese are downloaded from [FastText](https://fasttext.cc/docs/en/pretrained-vectors.html) (Wiki word vectors)
- Convertion of traditional-vs-simplified chinese characters is done by the python wrapper [OpenCC-Python](https://github.com/yichen0831/opencc-python) with respect to [Open Chinese Convert](https://github.com/BYVoid/OpenCC). 

### Model
- Link to the original paper: https://arxiv.org/abs/1707.01161

# Troubleshoot

### BLEU evaluation
- *OSError: [Errno 12] Cannot allocate memory*: Might not have enough RAM to run the .perl file. Try [adding a new swapfile](https://askubuntu.com/questions/927854/how-do-i-increase-the-size-of-swapfile-without-removing-it-in-the-terminal) (e.g. make to 4G total).


# Reference
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