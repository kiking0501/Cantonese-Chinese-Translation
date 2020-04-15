from configuration import DATA_PATH, MOVIE_PATH
import os
import numpy as np


def read_clean_transcript_pairs(name):
    data = []
    with open(os.path.join(MOVIE_PATH, "clean", "%s.original" % name)) as f1:
        with open(os.path.join(MOVIE_PATH, "clean", "%s.translate" % name)) as f2:
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                l1p, l2p = l1.partition(' '), l2.partition(' ')
                if l1p[0] != l2p[0]:
                    raise ValueError("(%s) Index is different for: (%s vs %s)!" % (name, l1, l2))
                data.append((l1p[2][:-1], l2p[2][:-1]))
    return data


def train_valid_split(data, train_ratio=0.8, test_num=500):
    ''' input: data(list)
               test_num(int): use None if do not want test data'''
    np.random.seed(2046)
    shuffle_ids = list(range(len(data)))
    np.random.shuffle(list(range(len(data))))

    if test_num is not None:
        test_ids = shuffle_ids[-test_num:]
        shuffle_ids = shuffle_ids[:-test_num]
    else:
        test_ids = None
    num_train = int((len(shuffle_ids)) * train_ratio)
    train_ids, valid_ids = shuffle_ids[:num_train], shuffle_ids[num_train:]

    if test_num is not None:
        return ([data[id_] for id_ in train_ids],
                [data[id_] for id_ in valid_ids],
                [data[id_] for id_ in test_ids])
    else:
        return ([data[id_] for id_ in train_ids],
                [data[id_] for id_ in valid_ids])


def read_data():
    data = []
    for ind in range(1, 8):
        data.extend(read_clean_transcript_pairs("0%s" % ind))
    return data
