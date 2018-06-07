import sys
import os
import time
from sssp.utils import utils
from sssp.config import exp_logger
from sssp.io.datasets import initDataset
from sssp.io.batch_iterator import threaded_generator
from sssp.utils.utils import average_res, res_to_string
import tensorflow as tf
import logging
import pickle as pkl
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# ------------- CHANGE CONFIGURATIONS HERE ---------------
conf_dirs = ['sssp.config.conf_clf',]
#conf_dirs = ['sssp.config.conf_clf_multilabel']
# --------------------------------------------------------

def validate(valid_dset, model, sess, idx2word, label2word, args):
    res_list = []
    threaded_it = threaded_generator(valid_dset, 200)
    f = open(os.path.join(args.save_dir, 'res_sample.txt'), 'w', encoding='utf-8')
    f.write('pred\ttarget\ttxt\n')
    for batch in threaded_it:
        prob, pred = model.classify(sess, *batch[:2])
        tgt = np.where(batch[2]==0, 0, batch[3]+1)
        pred = [label2word[idx] for idx in pred]
        tgt = [label2word[idx] for idx in tgt]
        inp = [[idx2word[idx] for idx in s if idx != 0] for s in batch[0]]
        
        for i in range(len(tgt)):
            f.write(pred[i] + '\t')
            f.write(tgt[i] + '\t')
            f.write(''.join(inp[i]) + '\n')

    return

def train_and_validate(args, model, sess, train_dset, valid_dset, test_dset, explogger):
    batch_cnt = 0
    res_list = []

    # init tensorboard writer
    tb_writer = tf.summary.FileWriter(args.save_dir + '/train', sess.graph)

    t_time = time.time()
    time_cnt = 0
    for epoch_idx in range(args.max_epoch):
        threaded_it = threaded_generator(train_dset, 200)
        for batch in threaded_it:
            batch_cnt += 1
            gen_time = time.time() - t_time
            t_time = time.time()
            res_dict, res_str, summary = model.run_batch(sess, batch, istrn=True)
            run_time = time.time() - t_time
            res_dict.update({'run_time': run_time})
            res_dict.update({'gen_time': gen_time})
            res_list.append(res_dict)
            res_list = res_list[-200:]
            time_cnt += gen_time + run_time

            if batch_cnt % args.show_every == 0:
                tb_writer.add_summary(summary, batch_cnt)
                out_str = res_to_string(average_res(res_list))
                explogger.message(out_str, True)
            
            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = validate(valid_dset, model, sess)
                explogger.message('VALIDATE: '  + out_str, True)

            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = validate(test_dset, model, sess)
                explogger.message('TEST: ' + out_str, True)

            if batch_cnt % args.save_every == 0:
                save_fn = os.path.join(args.save_dir, args.log_prefix)
                explogger.message("Saving checkpoint model: {} ......".format(args.save_dir))
                model.saver.save(sess, save_fn, 
                        write_meta_graph=False,
                        global_step=batch_cnt)

            t_time = time.time()

def main():
    # load all args for the experiment
    args = utils.load_argparse_args(conf_dirs=conf_dirs)
    explogger = exp_logger.ExpLogger(args.log_prefix, args.save_dir, write_file=False)
    wargs = vars(args)
    wargs['conf_dirs'] = conf_dirs
    explogger.write_args(wargs)
    explogger.file_copy(['sssp'])

    # step 1: import specified model
    module = __import__(args.model_path, fromlist=[args.model_name])
    model_class = module.__dict__[args.model_name]
    model = model_class(args)
    vt, vs = model.model_setup(args)
    explogger.write_variables(vs)

    # step 2: init dataset
    train_dset = initDataset(args.train_path, model.get_prepare_func(args), args.batch_size)
    valid_dset = initDataset(args.valid_path, model.get_prepare_func(args), 10)
    test_dset = initDataset(args.test_path, model.get_prepare_func(args), 10)

    vocab = pkl.load(open(args.vocab_path, 'rb'))
    idx2word = dict([[vocab[k], k] for k in vocab])
    labels = pkl.load(open(args.labels_path, 'rb'))
    label2word = dict([[labels[k]+1, k.strip()] for k in labels])
    label2word[0] = 'NAN'

    # step 3: Init tensorflow
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        model.saver.restore(sess, args.init_from)
        explogger.message('Model restored from {0}'.format(args.init_from))
        print(validate(test_dset, model, sess, idx2word, label2word, args))
        """
        train_and_validate(args, 
                model=model, 
                sess=sess, 
                train_dset=train_dset,
                valid_dset=valid_dset,
                test_dset=test_dset,
                explogger=explogger)
        """
