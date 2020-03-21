# Cantonese-Chinese-Translation

An experimental study on *Standard-Chinese* to *Cantonese* translator models.



Two major approaches are included:

- Copy-Enriched Seq2Seq Models (Jhamtani., 2017)
- Enriched dictionary table by Translation-Matrix (Mikolov., 2013)

This version of code is initiated from the work by <a href="https://github.com/harsh19/Shakespearizing-Modern-English" target="_blank">  Jhamtani </a>.



- [Requirements](#requirements)

- [Instructions to run](#instructions-to-run)

- [Details of Implementation](#details-of-implementation)

- [Troubleshoot](#troubleshoot)

- [Reference](#reference)

- [Changelog](#changelog)

  


# Requirements
- Python 3.5

- Tensorflow 1.1.0 (framework)

  

# Instructions to run:
Change working directory to **/code/v1/**

#### Initialization : 
- Run: </br>
`python mt_main.py init` </br>
to download the pre-trained Cantonese and Chinese embeddings from [fastText](https://fasttext.cc/docs/en/pretrained-vectors.html); and also to build token-dictionaries for the Chinese tokenizer

#### Preprocessing: 
- First run initialization
- Run: </br>
`python mt_main.py preprocessing` </br>
to pre-process and save train/valid/test data.
- The used dictionaries for tokenization can be changed at `prepro.py`

#### Copy-Enriched Seq-to-Seq Model: 
- First run pre-processing
- Run
`python mt_main.py train <iter_num> <output_model_name>` </br> or
`python mt_main.py <validation/test> <saved_model_name> <inference_type>`
- Training settings can be modified at `configuration.py`
- Trained models and results are saved in /data/tmp/
- Link to the original paper: https://arxiv.org/abs/1707.01161

#### Translation-Matrix Model: 
- First run pre-processing
- [Download](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh_yue.zip) and get the Cantonese language model (wiki.zh_yue.bin), put under data/embedding/
- Run
`python mt_translation_matrix.py` </br>
  and a linear projection matrix between the embedding spaces would be learnt using Stochastic Gradient Descent
- Trained matrix, internal validation accuracy and final results are saved in /data/translation_matrix /
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
- Conversion of traditional-vs-simplified Chinese characters is done by the python wrapper [OpenCC-Python](https://github.com/yichen0831/opencc-python) with respect to [Open Chinese Convert](https://github.com/BYVoid/OpenCC). 

### Evaluation
- BLEU metric evaluation is provided by the toolkit [MOSES](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)

- BLEU evaluation is adjusted to consider only single-Chinese-character tokenization

  


# Troubleshoot

### BLEU evaluation
- *OSError: [Errno 12] Cannot allocate memory*: Make sure one have enough RAM; restart the computer, or try [adding a new swapfile](https://askubuntu.com/questions/927854/how-do-i-increase-the-size-of-swapfile-without-removing-it-in-the-terminal) (e.g. make to 4G total).

  


# Reference
``` reStructuredText
- Keith Carlson, Allen Riddell, and Daniel Rockmore.  "Evaluating prose style transfer with the Bible". 2018. R. Soc. open sci. 5: 171829. http://dx.doi.org/10.1098/rsos.171920

- Francisco Guzman, Peng-Jen Chen, Myle OttF, Juan Pino, Guillaume Lample, Philipp Koehn, Vishrav Chaudhary,  Marc’Aurelio Ranzato. 2019. "The FLORES Evaluation Datasets for Low-Resource Machine Translation: Nepali–English and Sinhala–English", arXiv:1902.01382v3, EMNLP 2019.

- Harsh Jhamtani, Varun Gangal, Eduard Hovy, and Eric Nyberg. 2017. "Shakespearizing modern language using copy-enriched sequence-to-sequence models." Proceedings of the Workshop on Stylistic Variation, EMNLP 2017. 

- Huang G, Gorin A, Gauvain JL, Lamel L. Machine translation based data augmentation for Cantonese keyword spotting. In2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2016 Mar 20 (pp. 6020-6024). IEEE.

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781.

- Jackson L. Lee, Litong Chen, and Tsz-Him Tsui. 2016. PyCantonese: Developing computational tools for Cantonese linguistics. Talk at the 3rd Workshop on Innovations in Cantonese Linguistics, The Ohio State University. March 12. 2016.

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002.  "Bleu: a method for automatic evaluation of machine translation." In Proceedings of
  the 40th annual meeting on association for computational linguistics. Association for Computational
  Linguistics, pages 311–318.

- Mikolov, Tomas, Quoc V. Le, and Ilya Sutskever. "Exploiting similarities among languages for machine translation." arXiv preprint arXiv:1309.4168 (2013).

- Rao S, Tetreault J. 2018.  "Dear sir or madam, may I introduce the YAFC corpus: corpus, benchmarks and metrics for formality style transfer." In Proceeding of the 2018 Conf. of the North American chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1, New Orleans, LA, June, pp. 129– 140. Association for Computational Linguistics. (doi:10.18653/v1/n18-1012)

- Wong, Tak-sum, and John Lee. "Register-sensitive Translation: a Case Study of Mandarin and Cantonese (Non-archival Extended Abstract)." Proceedings of the 13th Conference of the Association for Machine Translation in the Americas (Volume 1: Research Papers). 2018.

- Mengzhou Xia, Xiang Kong, Antonios Anastasopoulos, Graham Neubig. 2019. "Generalized Data Augmentation for Low-Resource Translation", arXiv:1906.03785, ACL 2019

- Wei Xu, Alan Ritter, Bill Dolan, Ralph Grishman, and Colin Cherry. 2012. "Paraphrasing for style". Proceedings of COLING 2012 pages 2899–2914

- Xu, Jia, Richard Zens, and Hermann Ney. "Do we need Chinese word segmentation for statistical machine translation?." Proceedings of the third SIGHAN workshop on Chinese language processing. 2004.

- Chunting Zhou, Xuezhe Ma, Junjie Hu, Graham Neubig. 2019. "Handling Syntactic Divergence in Low-resource Machine Translation", arXiv:1909.00040v1, EMNLP 2019.

- Zhang, Xiaoheng. "Dialect MT: a case study between Cantonese and Mandarin." Proceedings of the 36th Annual Meeting of the Association for Computational Linguistics and 17th International Conference on Computational Linguistics-Volume 2. Association for Computational Linguistics, 1998.

- Bojanowski, Piotr, et al. "Enriching word vectors with subword information." Transactions of the Association for Computational Linguistics 5 (2017): 135-146.

```



# Changelog

### 2020-Mar

- Added reference and improved README.  
- Restart the project. Rename /code/main to /code/v1 



### 2019-Dec

- Summarized the result with a report titled "**Dialect as a Low-Resource Language: A Study on Standard-Chinese to Cantonese Translation with Movie Transcripts**" . Abstract:

  > Cantonese, a major Chinese spoken dialect, can be viewed a a low-resource language given that its raw written form of collection is scarce. This project develops a pipeline to accomplish the low-resource Cantonese translation task with its closely-related rich-resource language counterparts, Standard Chinese (SC). The pipeline consists of two major translation methods: (1) the sequence-to-sequence neural-network approach suggested by Jhamtani et al. (2017), and (2) the translation-matrix approach suggested by Mikolov et al. (2013). Our implementation to perform machine translation from SC to Cantonese, in a simplified setting, do not have satisfying results nor perform better than the baselines. This report describes the similarities and difference between our implementation and the original approaches, and also discusses possible future improvement.
  
  

-  Submitted the report for the UWaterloo course CS680 - Introduction to Machine Learning. Grade 21/25. 

  

### 2019-Oct

- Initialized the repository



