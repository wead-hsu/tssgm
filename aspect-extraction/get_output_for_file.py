from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

from model.data_utils import minibatches, get_chunks

# -*- coding: utf-8 -*
import os
#os.system('pip install jieba')
os.system('cp -r nltk_data/ ~/')

import tensorflow as tf
import numpy as np
#import data_helper
import math
import re
import odps_config
import env_processor
import odps_writer_partition
#import jieba
import sys
import itertools
import multiprocessing
from nltk.tokenize import word_tokenize as wt

'''
def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


'''

#print(os.path.exists('results/test/model.weights/-0.data-00000-of-00001'))
print(os.path.exists('~/nltk_data/tokenizers/'))
print(os.path.exists('/usr/local/share/nltk_data/'))
print(os.path.exists('/usr/local/share/nltk_data/tokenizers'))
print('fsfdsfs')
config = Config()


def data_iterator(block_id, block_num):
    odps_data_params = odps_config.odps_data_params.copy()
    #odps_data_params['partition']='dt=20180506'
    data_count = env_processor.make_env(odps_data_params).count()
    per_thread_size = data_count / block_num
    range_start = per_thread_size * block_id
    range_end = min(per_thread_size * (block_id + 1), data_count)
    odps_data_params.update({"range_start": range_start, "range_end":range_end})
    sample_iter = env_processor.make_env(odps_data_params)

    #odps_results_params['partition']='dt=20180506'

    #result_writer = odps_writer_partition.make_writer(odps_results_params)
    data_size = range_end - range_start
    #VOCAB_SIZE = word2vector_helpers.load_vocab_size(FLAGS.embedding_dim)
    #pretrained_word2vec_matrix = word2vector_helpers.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)
    i = 0
    pos = 0
    while pos < data_size:
        print pos, data_size
        bs = min(config.batch_size, data_size - pos)
        sample = list(itertools.islice(sample_iter, pos, pos + bs))
        #print(bs)
        #print(type(sample))
        #print(sample[0])
        #print(sample[0][0]['comment_txt'])
        comment = [x[0]['comment_txt'] for x in sample]
        tokens = [wt(x[0]['comment_txt']) for x in sample]
        words = [zip(*[config.processing_word(w) for w in x]) for x in tokens]
        tags = [[config.processing_tag('B') for w in x] for x in tokens]
        #print(words, tags)
        #print(model.predict_batch(words))
        yield words, tags, tokens, comment
        if len(sample) == 0:
            print "sample's length is zero"
            i += 1
            pos += config.batch_size
            continue
        i += 1
        pos += config.batch_size

def main(block_id, block_num):
    # create instance of config
    #config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.saver.restore(model.sess, './results/test/model.weights/-0')
    model.restore_session('./results/test/model.weights/-0')
    model_file = tf.train.latest_checkpoint("./results/test/model.weights")

    # create dataset
    #test  = CoNLLDataset(config.filename_test, config.processing_word,
                         #config.processing_tag, config.max_iter)
    it = data_iterator(block_id, block_num)

    # evaluate and interact
    #model.evaluate(test)
    #interactive_shell(model)

    idx2word = {config.vocab_words[k]: k for k in config.vocab_words}
    idx2tag = {config.vocab_tags[k]: k for k in config.vocab_tags}

    odps_results_params = odps_config.odps_results_params.copy()
    result_writer = odps_writer_partition.make_writer(odps_results_params)

    line_cnt = 0
    for words, labels, tokens, comments in it:
        labels_pred, sequence_lengths = model.predict_batch(words)
        records = []
        for comment, token, x, lab, lab_pred, length in zip(comments, tokens, words, labels, labels_pred,
                                         sequence_lengths):
            line_cnt += 1
            if line_cnt % 100 == 0:
                print('block id {}: {}'.format(block_id, line_cnt))

            aspects = []
            #print(len(x))
            lab_pred = lab_pred[:length]
            #print(x[1], lab, lab_pred)
            
            idx = 0
            while idx < length:
                aspect = []
                while idx < length and lab_pred[idx] != 2:
                    #aspect.append(idx2word[x[1][idx]])
                    aspect.append(token[idx])
                    #print(token[idx])
                    idx += 1
                if len(aspect) != 0:
                    aspects.append(' '.join(aspect))
                idx += 1
            aspects = '\t'.join(aspects)
            tags = '\t'.join([idx2tag[t] for t in lab_pred])
            #print(comment,tags,aspects)
            #print(lab_pred)
            records.append([comment, tags, aspects])
        result_writer.write_record(records, block_id)

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    predict_process = []
    for i in range(cpu_count - 1):
        p = multiprocessing.Process(target=lambda: main(i, cpu_count - 1))
        p.start()
        predict_process.append(p)

    for p in predict_process:
        p.join()

