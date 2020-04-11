import configuration as config
import os

import tensorflow as tf
import numpy as np
# Set seed for reproducability
tf.set_random_seed(1)
np.random.seed(1)

if config.use_gpu:
    # limit GPU usage
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.preprocessing.sequence import pad_sequences

import pickle
import sys
import utilities as datasets
import utilities
import dao
import mt_solver as solver
from prepro import PreProcessing, modifyParamsWithPrepro
import json

########################
usage = '''
python mt_main.py train <num_of_iters> <model_name>
OR
python mt_main.py validation <saved_model_name> <greedy/beam>
OR
python mt_main.py test <saved_model_name> <greedy/beam>
OR
python mt_main.py debug
OR
python mt_main.py preprocessing
OR
python mt_main.py init
'''
print ("USAGE: ")
print (usage)
print ("")
########################

data_dir = config.data_dir
save_dir = config.SAVE_PATH
transcript_files = config.transcript_files


def main():

    # params
    params = config.params

    mode = sys.argv[1]
    print ("mode = ", mode)

    if mode not in ['validation', 'test']:
        print("PARAMS:")
        for key, value in params.items():
            print (" -- ", key, " = ", value)
        buckets = {0: {'max_input_seq_length': params['max_input_seq_length'], 'max_output_seq_length': params['max_output_seq_length']} }
        print ("buckets = ", buckets)


    ########### INIT
    if mode == "init":
        dao.translate_jieba_dict()
        dao.create_jieba_dict_pycanto()
        if not (os.path.exists(params['canto_embedding_path']) or
                os.path.exists(params['stdch_embedding_path'])):
            dao.download_embedding()
            dao.save_embedding("cantonese/wiki.zh_yue.vec", "canto_wiki.pkl")
            dao.save_embedding("standard_chinese/wiki.zh.vec", "stdch_wiki.pkl")
        dao.merge_dict_txt_with_embedding_tokens("dict.txt.pycanto", "canto_wiki.pkl")
        return

    ########### PREPROCESSING

    if mode == "preprocessing":
        trans_data = []
        for trans_file in transcript_files:
            trans_data += dao.read_clean_transcript_pairs(trans_file)
        dao.save_train_valid(*dao.train_valid_split(trans_data))

        # preprocesing
        print("=" * 5)
        preprocessing = PreProcessing()
        splits = ["train", "valid", "test"]
        preprocessing.loadVocab('train')
        if params['do_vocab_pruning']:
            preprocessing.pruneVocab(max_vocab_size=params['max_vocab_size'])
        data_seq = {split: preprocessing.loadData(split=split) for split in splits}
        data = {split: preprocessing.prepareMTData(cur_data) for split, cur_data in data_seq.items()}
        pickle.dump(data, open(os.path.join(data_dir, "data.obj"), "wb"))
        pickle.dump(preprocessing, open(os.path.join(data_dir, "preprocessing.obj"), "wb"))
        return

    else:
        data = pickle.load(open(data_dir + "data.obj", "rb"))
        preprocessing = pickle.load(open(data_dir + "preprocessing.obj", "rb"))

    train = data['train']
    val = data['valid']
    test = data['test']

    # DEBUG
    if mode == "debug":
        lim = 64
    else:
        lim = params['batch_size'] * int(len(train[0])/params['batch_size'])

    if lim != -1:
        train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, train_decoder_outputs_matching_inputs = train
        train_encoder_inputs = train_encoder_inputs[:lim]
        train_decoder_inputs = train_decoder_inputs[:lim]
        train_decoder_outputs = train_decoder_outputs[:lim]
        train_decoder_outputs_matching_inputs = train_decoder_outputs_matching_inputs[:lim]
        train = train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, train_decoder_outputs_matching_inputs

    # TRAIN/DEBUG
    if mode == 'train' or mode == "debug":
        if mode == "train":
            training_iters = int(sys.argv[2])
            model_name = sys.argv[3]
        else:
            training_iters = 5
            model_name = "test"
        params['training_iters'] = training_iters
        params['model_name'] = model_name
        train_buckets = {}
        for bucket, _ in enumerate(buckets):
            train_buckets[bucket] = train

        json.dump(params, open(os.path.join(save_dir, "%s.params" % model_name), "w"))
        params = modifyParamsWithPrepro(params, preprocessing)
        rnn_model = solver.Solver(params, buckets)
        _ = rnn_model.getModel(params, mode='train', reuse=False, buckets=buckets)
        rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=val, reverse_vocab=preprocessing.idx_to_word, do_init=True)

    # Validation
    elif mode == "validation":
        saved_model_path = os.path.join(save_dir, sys.argv[2])
        print ("saved_model_path = ",saved_model_path)

        inference_type = sys.argv[3] # greedy / beam
        print( "inference_type = ",inference_type)

        params_path = os.path.join(data_dir, "%s.params" % saved_model_path.rpartition('/')[2].partition('.')[0])
        if os.path.exists(params_path):
            params = json.load(params)
            print("successfully load params.")

        params['saved_model_path'] = saved_model_path
        params = modifyParamsWithPrepro(params, preprocessing)
        rnn_model = solver.Solver(params, buckets=None, mode='inference')
        _ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
        print ("----Running on Validation Set-----")

        #val
        val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, val_decoder_outputs_matching_inputs = val
        #print "val_encoder_inputs = ",val_encoder_inputs
        if len(val_decoder_outputs.shape)==3:
            val_decoder_outputs=np.reshape(val_decoder_outputs, (val_decoder_outputs.shape[0], val_decoder_outputs.shape[1]))
        decoder_outputs_inference, decoder_ground_truth_outputs = rnn_model.solveAll(
            params, val_encoder_inputs, val_decoder_outputs, preprocessing.idx_to_word,
            inference_type=inference_type, print_progress=False)

        model_name = saved_model_path.rpartition('/')[2]
        validOutFile_name = os.path.join(save_dir, model_name + ".valid.output")
        original_data_path = data_dir + preprocessing.OUT_SRC['valid'] + '.tok.char'
        BLEUOutputFile_path = os.path.join(save_dir, model_name + ".valid.BLEU")
        utilities.getBlue(
            validOutFile_name, original_data_path, BLEUOutputFile_path,
            decoder_outputs_inference, decoder_ground_truth_outputs,
            preprocessing)

        print ("VALIDATION: ",open(BLEUOutputFile_path,"r").read())

    # TEST
    elif mode == "test":
        saved_model_path = os.path.join(save_dir, sys.argv[2])
        print ("saved_model_path = ",saved_model_path)

        inference_type = sys.argv[3] # greedy / beam
        print ("inference_type = ",inference_type)

        params_path = os.path.join(data_dir, "%s.params" % saved_model_path.rpartition('/')[2].partition('.')[0])
        if os.path.exists(params_path):
            params = json.load(params)
            print("successfully load params.")

        params['saved_model_path'] = saved_model_path
        params = modifyParamsWithPrepro(params, preprocessing)
        rnn_model = solver.Solver(params, buckets=None, mode='inference')
        _ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
        print ("----Running on Test Set-----")

        test_encoder_inputs, test_decoder_inputs, test_decoder_outputs, test_decoder_outputs_matching_inputs = test
        if len(test_decoder_outputs.shape)==3:
            test_decoder_outputs=np.reshape(test_decoder_outputs, (test_decoder_outputs.shape[0], test_decoder_outputs.shape[1]))
        decoder_outputs_inference, decoder_ground_truth_outputs = rnn_model.solveAll(
            params, test_encoder_inputs, test_decoder_outputs, preprocessing.idx_to_word,
            inference_type=inference_type, print_progress=False)

        model_name = saved_model_path.rpartition('/')[2]
        validOutFile_name = os.path.join(save_dir, model_name + ".test.output")
        original_data_path = data_dir + preprocessing.OUT_SRC['test'] + '.tok.char'
        BLEUOutputFile_path = os.path.join(save_dir, model_name + ".test.BLEU")
        utilities.getBlue(
            validOutFile_name, original_data_path, BLEUOutputFile_path,
            decoder_outputs_inference, decoder_ground_truth_outputs,
            preprocessing)

        print ("TEST: ",open(BLEUOutputFile_path,"r").read())

    else:
        print ("Please see usage")


if __name__ == "__main__":
    main()
