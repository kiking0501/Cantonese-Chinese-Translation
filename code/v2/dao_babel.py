import os
from opencc import OpenCC
from configuration import BABEL_PATH, BABEL_ORI_PATH
from collections import defaultdict


if not os.path.exists(os.path.join(BABEL_ORI_PATH, "conversational")):
    raise IOError("You need to download IARPA_BABEL_BP_101_LDC2016S02.zip from\n"
                  " https://catalog.ldc.upenn.edu/LDC2016S02\n"
                  " extract and place the complete IARPA_BABEL_BP_101 folder\n"
                  " under %s!" % BABEL_PATH)

s2t = OpenCC('s2t')


def read_conversation(file_code, folder_path):
    ''' conversation: inLine + outLine '''
    d = {}
    for suffix in ("_inLine.txt", "_outLine.txt"):
        file_path = os.path.join(folder_path, file_code + suffix)
        if not os.path.exists(file_path):
            x1, x2, x3 = file_path.rpartition('_')
            file_path = "%s%s%s%s" % (x1[:-1], int(x1[-1]) + 1, x2, x3)
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                time = lines[i].strip()
                l = lines[i+1].strip() if (i+1 < len(lines)) else None
                d[time] = l
    return d


def save_clean_conversation(file_code, folder_path, verbose=True):
    conversation_dict = read_conversation(file_code, folder_path)
    output_path = os.path.join(BABEL_PATH, "clean", file_code)
    if conversation_dict:
        with open(output_path, "w") as f:
            for time, l in sorted(conversation_dict.items(), key=lambda x: float(x[0][1:-1])):
                line = s2t.convert(l) if l is not None else None
                if line:
                    # maybe rewrite with re later..
                    tokens = line.replace('<no-speech>', ',').replace('<laugh>', ',').strip(' ,').split(' ')
                    tokens = [t.strip('-') for t in tokens if not t.startswith('<') and not t.startswith('(') and t]
                    tokens = [t for t in ' '.join(tokens).replace(', ,', ',').split(' ') if t]
                    if tokens:
                        f.write(' '.join(tokens) + '\n')
        if verbose:
            print("%s saved." % output_path)
    return conversation_dict


def read_clean_conversation(file_code):
    with open(os.path.join(BABEL_PATH, "clean", file_code)) as f:
        return [l.strip().split(' ') for l in f]


def create_transcription_overview_csv(output_file="transcription_overview.csv", verbose=True):
    ''' transcription: a collection of conversations '''
    total = 0
    with open(os.path.join(BABEL_PATH, output_file), "w") as f:
        for folder_name in ["dev", "sub-train", "training"]:
            files = []
            for (_, _, filenames) in os.walk(os.path.join(BABEL_ORI_PATH, "conversational", folder_name, "transcription")):
                files.extend(filenames)
            file_codes = set([x.rpartition('_')[0] for x in files])
            for ind, code in enumerate(sorted(file_codes)):
                f.write('%s,%s,%s,%s\n' % (ind, code, "conversational", folder_name))
            total += len(file_codes)
    if verbose:
        print("Total: %d. %s saved." % (total, output_file))


def save_clean_transcription(overview_csv="transcription_overview.csv", verbose=True):
    saved = defaultdict(lambda: defaultdict(int))
    with open(os.path.join(BABEL_PATH, overview_csv)) as f:
        for l in f.readlines():
            _, file_code, type_, sub_type = l.strip().split(',')
            folder_path = os.path.join(BABEL_ORI_PATH, type_, sub_type, "transcription")
            save_clean_conversation(file_code, folder_path)
            saved[file_code][sub_type] += 1

    print("Total: %d saved." % (len(saved)))

    saved_txt = os.path.join(BABEL_PATH, "%s_saved.txt" % overview_csv.rpartition('.')[0])
    saved_lines = [k for k, v in sorted(saved.items(), key=lambda x: x[0])]
    with open(saved_txt, "w") as f:
        for l in saved_lines:
            f.write("%s\n" % l)
    print("%s saved." % saved_txt)

    repeated_files = [(k, v) for k, v in sorted(saved.items(), key=lambda x: -sum(x[1].values()))
                      if sum(v.values()) > 1]
    print("Repeated: %d" % (len(repeated_files)))
    if verbose:
        for k, v in repeated_files:
            print("%s: {%s}" % (k, ", ".join(["%s: %s" % (v1, v2) for v1, v2 in v.items()])))
    return saved_lines, repeated_files


def save_clean_transcription_combine(output_file="transcription_combine.txt", overview_txt="transcription_overview_saved.txt"):
    transcription = []

    with open(os.path.join(BABEL_PATH, overview_txt)) as f:
        file_code_list = [l.strip() for l in f.readlines()]

    for file_code in file_code_list:
        transcription.extend(read_clean_conversation(file_code))

    with open(os.path.join(BABEL_PATH, output_file), "w") as f:
        for t in transcription:
            f.write(' '.join(t) + '\n')
    print("%s saved." % output_file)
    return transcription


def read_clean_transcription_combine(combine_txt="transcription_combine.txt"):
    with open(os.path.join(BABEL_PATH, combine_txt)) as f:
        corpus = [l.strip().split(' ') for l in f.readlines()]
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
