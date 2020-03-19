

# Train a Cantonese Embedding using the Cantonese-Wikipedia

### 1. Download the Cantonese-Wikipedia

- The Wikipedia database dump ver. 20200301:  https://dumps.wikimedia.org/zh_yuewiki/20200301/ 
- The dump we are using here is `zh_yuewiki-20200301-pages-articles-multistream.xml.bz2` (57.8MB)

### 2. Download <a href="https://github.com/attardi/wikiextractor"> WikiExtractor </a> 

- `WikiExtractor.py` is a script that extracts and cleans text from a Wikipedia database dump. 

  Download link: https://github.com/attardi/wikiextractor/archive/master.zip

- After decompression of the zip file, go to the wikiextractor directory, run in terminal

  `WikiExtractor.py -o <OUTPUT directory> --json -b 500K <path-to-your-dump-file>`

  where
  `-o <OUTPUT directory>` specifies the output directory

  `--json` writes output in json format instead of the default one

  `-b 500K` specifies the maximum bytes per output file (default 1M)

  `<path-to-your-dump-file` is the path of the .xml.bz2 file

- The terminal should end with the logging message similar to the below: 

  ```bash
  INFO: Finished 7-process extraction of 77685 articles in 41.1s (1888.5 art/s)
  INFO: total of page: 114455, total of articl page: 77685; total of used articl page: 77685
  ```

- The extracted files are stored in the `<OUTPUT directory>` (under an `AA` directory?) 

  An example extracted file (`wiki_00`) will look like: 

  ```json
  {"url": "https://zh-yue.wikipedia.org/wiki?curid=1", "text": "頭版/2013\n", "id": "1", "title": "頭版/2013"}
  {"url": "https://zh-yue.wikipedia.org/wiki?curid=2", "text": "香港\n\n香港（，），係華南一城，亦係一埠，譽為國際大都會。...... 常見嘅街頭小食有雞蛋仔、蛋撻、咖喱魚蛋、燒賣、格仔餅等等，嘢飲就有絲襪奶茶，珍珠奶茶等等。\n", "id": "2", "title": "香港"}
  ......
  ```

  Each line is an article stored in the json format. An extracted file will contain several articles (json), depends on the specified maximum bytes per output file.   

### 3. Prepare the training corpus

- Gensim's word2vec expects a sequence of sentences as its input.  Each sentence should be a list of tokens.

- To prepare the training corpus, we need to cut the article texts into sentences, followed by a tokenization.

- Example code:

  ```python
  import jieba
  import os
  
  def load_jieba(dict_txt="dict.txt.pycanto-stdch_wiki", verbose=False):
      ''' load jieba with the stated dictionary txt (valid after ver 0.28) '''
      jieba.set_dictionary(dict_txt)
      if verbose:
          print("Successfully set Jieba-dict to '%s'. " % dict_txt)
  
  def read_wiki(file_name):
      ''' read json object from an extracted wiki file '''
      data = []
      with open(os.path.join(<OUTPUT directory>, file_name), 'r') as f:
          for json_obj in f:
              data.append(json.loads(json_obj))
      return data
  
  def read_wiki_sentences(file_name, verbose=True):
      ''' Very simple way to cut the wiki article into sentences,
          followed by jieba tokenization '''
      json_list = read_wiki(file_name)
      for json_obj in json_list:
          for line in json_obj['text'].split('\n'):
          	for l in line.split('。'):
                 	if l:
                     yield jieba.cut(l, cut_all=False) + ['。']
  
  if __name__ == '__main__':
      load_jieba()
      for sentence in read_wiki_sentences("wiki_00"):
          print(sentence)
  ```

- The result would look something like

  ```python
  ["頭版", "/", "2013", "。"]
  ["香港", "。"]
  ["香港", "，", "係", "華南", "一城"， "亦", "係", "一埠", "，", "譽為", "國際", "大都", "會", "。"]
  ......
  ```

### 4. Train the word embedding using Gensim's word2vec

- Basically follow this tutorial: https://rare-technologies.com/word2vec-tutorial/

- Example Training Code:

  ```python
  from gensim.models import Word2Vec
  import time
  import logging
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  
  FILES = # ["wiki_00", "wiki_01", ...]
  
  class CantoSentences(object):
      def __init__(self):
          pass
  
      def __iter__(self):
      	for file_name in FILES:
          	for sentence in read_wiki_sentences(file_name):
                  yield sen
  
  if __name__ == '__main__':
      st = time.time()
  
      corpus = CantoSentences()
      model = Word2Vec(corpus, size=300, min_count=5, iter=5)
  
      print("Training time: %s mins." % ((time.time() - st) / 60))
  
      model.save("wiki_canto.model")
      model.wv.save_word2vec_format('wiki_canto.bin', binary=True)
  ```

- In the actual training, the iteration is set to 50 instead and takes around 30 minutes to finish (fast!). 

- Example Code to load the trained model:

  ```python
  from gensim.models import Word2Vec
  model = Word2Vec.load("wiki_canto.model")
  ```

- Example result:

  ```python
  model.wv.similar_by_word("廣東話")
  ```

  ```
  [('粵語', 0.5983456373214722),
   ('廣州話', 0.4517979025840759),
   ('普通話', 0.44722360372543335),
   ('官話', 0.43270301818847656),
   ('客家話', 0.4313108026981354),
   ('口音', 0.4249879717826843),
   ('閩南話', 0.4159209430217743),
   ('廣東話。', 0.4106157720088959),
   ('詞彙', 0.4050295054912567),
   ('中文', 0.4048430323600769)]
  ```