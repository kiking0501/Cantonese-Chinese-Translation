import os
from bs4 import BeautifulSoup
from collections import defaultdict
from configuration import HKCANCOR_PATH, HKCANCOR_ORI_PATH


if not os.path.exists(HKCANCOR_ORI_PATH):
    raise IOError("You need to download hkcancor-utf8.zip from\n"
                  " https://github.com/fcbond/hkcancor/tree/master/data\n"
                  " extract and place the complete hkcancor-utf8 folder\n"
                  " under %s!" % HKCANCOR_PATH)


def read_conversation(file_code):
    with open(os.path.join(HKCANCOR_ORI_PATH, "FC-001_v2")) as f:
        soup = BeautifulSoup(f.read())
    sent_list = []
    for sent_tag in soup.findAll('sent_tag'):
        sent_list.append([t.strip('\t').partition('/')[0] for t in sent_tag.contents[0].split('\n\t\t') if t])
    return sent_list


def save_clean_conversation(file_code, verbose=True):
    sent_list = read_conversation(file_code)
    output_path = os.path.join(HKCANCOR_PATH, "clean", file_code)
    with open(output_path, "w") as f:
        for tokens in sent_list:
            f.write(' '.join(tokens) + '\n')
    if verbose:
        print("%s saved." % output_path)
    return sent_list


def read_clean_conversation(file_code):
    with open(os.path.join(HKCANCOR_PATH, "clean", file_code)) as f:
        return [l.strip().split(' ') for l in f]


def create_transcription_overview_csv(output_file="transcription_overview.csv", verbose=True):
    ''' transcription: a collection of conversations '''
    total = 0
    with open(os.path.join(HKCANCOR_PATH, output_file), "w") as f:
        file_codes = []
        for (_, _, filenames) in os.walk(HKCANCOR_ORI_PATH):
            file_codes.extend(filenames)
        for ind, code in enumerate(sorted(file_codes)):
            f.write('%s,%s\n' % (ind, code))
        total += len(file_codes)
    if verbose:
        print("Total: %d. %s saved." % (total, output_file))


def save_clean_transcription(overview_csv="transcription_overview.csv", verbose=True):
    saved = defaultdict(lambda: defaultdict(int))
    with open(os.path.join(HKCANCOR_PATH, overview_csv)) as f:
        for l in f.readlines():
            _, file_code = l.strip().split(',')
            save_clean_conversation(file_code)

    print("Total: %d saved." % (len(saved)))


def save_clean_transcription_combine(output_file="transcription_combine.txt", overview_csv="transcription_overview.csv"):
    transcription = []

    with open(os.path.join(HKCANCOR_PATH, overview_csv)) as f:
        file_code_list = [l.split(',')[1].strip() for l in f.readlines()]

    for file_code in file_code_list:
        transcription.extend(read_clean_conversation(file_code))

    with open(os.path.join(HKCANCOR_PATH, output_file), "w") as f:
        for t in transcription:
            f.write(' '.join(t) + '\n')
    print("%s saved." % output_file)
    return transcription


def read_clean_transcription_combine(combine_txt="transcription_combine.txt", tokenize=True):
    with open(os.path.join(HKCANCOR_PATH, combine_txt)) as f:
        corpus = [l.strip().split(' ') if tokenize else l.strip().replace(' ', '') for l in f.readlines()]
    return corpus


if __name__ == '__main__':
    create_transcription_overview_csv()
    save_clean_transcription()
    save_clean_transcription_combine()
    corpus = read_clean_transcription_combine()

    tokdict = defaultdict(int)
    for sen in corpus:
        for w in sen:
            tokdict[w] += 1
    for k, v in sorted(tokdict.items(), key=lambda x: -x[1])[:100]:
        print(k, v)
