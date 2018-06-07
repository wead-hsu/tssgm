# load difference flags configurations from 
# different files

import tensorflow as tf
import numpy as np
import argparse

tf.app.flags.DEFINE_string('dummy_constant', '-', '')

def load_tensorflow_args(conf_dirs):
    for cdir in conf_dirs:
        __import__(cdir)

    flags = tf.app.flags.FLAGS
    # Hard code here:
    # 1. get one arbitrary attribute in the flags
    #   to let the parser parse the arguments
    flags.dummy_constant
    return flags

def load_argparse_args(conf_dirs):
    parser = argparse.ArgumentParser()
    for cdir in conf_dirs:
        module = __import__(cdir, fromlist=['init_arguments'])
        f = module.__dict__['init_arguments']
        f(parser)
    
    flags = parser.parse_args()
    return flags

def res_to_string(res_dict):
    res_str = ''
    for k in sorted(res_dict.keys()):
        res_str += '{0}: {1:0.3f}'.format(k, res_dict[k]) + '\t'
    return res_str

def average_res(res_list):
    if len(res_list) == 0:
        return {}
    keys = res_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = np.mean([res[k] for res in res_list])
    return avg

def confusion_matrix_to_string(matrix, labels):
    res = '\t'
    res += '\t'.join(labels) + '\n'
    for idx, l in enumerate(labels):
        res += l + '\t' + '\t'.join([str(f) for f in matrix[idx]]) + '\n'
    return res

def get_prepare_clf_func_help(args):
    func = get_prepare_func
    if args.task_id is not None: func = get_prepare_func_for_certain_task
    return func

def get_prepare_func_for_certain_task(args):
    num_tasks = len(args.list_num_classes.split(','))
    def prepare_data(raw_inp):
        raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
        raw_inp = list(zip(*raw_inp))
        labels = raw_inp[:num_tasks]
        label = labels[args.task_id]
        inp = raw_inp[num_tasks]
        
        def proc(sents):
            sent_lens = [len(s) for s in sents]
            max_sent_len = min(args.max_sent_len, max(sent_lens))
            if args.fix_sent_len > 0: max_sent_len = args.fix_sent_len - 1
            batch_size = len(sents)
            inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
            tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
            msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
            for idx, s in enumerate(sents):
                len_s = min(max_sent_len, len(s))
                inp_np[idx][1:len_s+1] = s[:len_s]
                tgt_np[idx][:len_s] = s[:len_s]
                msk_np[idx][:len_s+1] = 1
            return inp_np, tgt_np, msk_np
        
        inp = proc(inp)
        inp = (inp[1], inp[2])
        label = np.asarray(label).flatten().astype('int32')

        return inp + (label,)
    return prepare_data

def get_prepare_func(args):
    def prepare_data(raw_inp):
        raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
        raw_inp = list(zip(*raw_inp))
        labels = raw_inp[:-1]
        inp = raw_inp[-1]
        
        def proc(sents):
            sent_lens = [len(s) for s in sents]
            max_sent_len = min(args.max_sent_len, max(sent_lens))
            if args.fix_sent_len > 0: max_sent_len = args.fix_sent_len - 1
            batch_size = len(sents)
            inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
            tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
            msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
            for idx, s in enumerate(sents):
                len_s = min(max_sent_len, len(s))
                inp_np[idx][1:len_s+1] = s[:len_s]
                tgt_np[idx][:len_s] = s[:len_s]
                msk_np[idx][:len_s+1] = 1
            return inp_np, tgt_np, msk_np
        
        inp = proc(inp)
        inp = [inp[1], inp[2]]
        labels = [np.asarray(label).flatten().astype('int64') for label in labels]
        return inp + labels
    return prepare_data

def get_prepare_reg_func_for_certain_task(args):
    num_tasks = len(args.list_num_classes.split(','))
    def prepare_data(raw_inp):
        raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
        raw_inp = list(zip(*raw_inp))
        labels = raw_inp[:num_tasks]
        label = labels[args.task_id]
        inp = raw_inp[num_tasks]
        
        def proc(sents):
            sent_lens = [len(s) for s in sents]
            max_sent_len = min(args.max_sent_len, max(sent_lens))
            if args.fix_sent_len > 0: max_sent_len = args.fix_sent_len - 1
            batch_size = len(sents)
            inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
            tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int32')
            msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
            for idx, s in enumerate(sents):
                len_s = min(max_sent_len, len(s))
                inp_np[idx][1:len_s+1] = s[:len_s]
                tgt_np[idx][:len_s] = s[:len_s]
                msk_np[idx][:len_s+1] = 1
            return inp_np, tgt_np, msk_np
        
        inp = proc(inp)
        inp = (inp[1], inp[2])
        label = np.asarray(label).flatten().astype('float32')

        return inp + (label,)
    return prepare_data

def get_prepare_reg_func(args):
    def prepare_data(raw_inp):
        raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
        raw_inp = list(zip(*raw_inp))
        labels = raw_inp[:-1]
        inp = raw_inp[-1]
        
        def proc(sents):
            sent_lens = [len(s) for s in sents]
            max_sent_len = min(args.max_sent_len, max(sent_lens))
            if args.fix_sent_len > 0: max_sent_len = args.fix_sent_len
            batch_size = len(sents)
            inp_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
            tgt_np = np.zeros([batch_size, max_sent_len+1], dtype='int64')
            msk_np = np.zeros([batch_size, max_sent_len+1], dtype='float32')
            for idx, s in enumerate(sents):
                len_s = min(max_sent_len, len(s))
                inp_np[idx][1:len_s+1] = s[:len_s]
                tgt_np[idx][:len_s] = s[:len_s]
                msk_np[idx][:len_s+1] = 1
            return inp_np, tgt_np, msk_np
        
        inp = proc(inp)
        inp = [inp[1], inp[2]]
        labels = [np.asarray(label).flatten().astype('float32') for label in labels]
        return inp + labels
    return prepare_data

