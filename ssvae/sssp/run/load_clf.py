import sys
import os
import time
import numpy as np
import pickle as pkl
from sssp.utils import utils
from sssp.config import exp_logger
from sssp.io.datasets import initDataset
from sssp.io.batch_iterator import threaded_generator
from sssp.utils.utils import average_res, res_to_string
import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)

# ------------- CHANGE CONFIGURATIONS HERE ---------------
conf_dirs = ['sssp.config.conf_clf',]
#conf_dirs = ['sssp.config.conf_clf_multilabel']
# --------------------------------------------------------

def validate(valid_dset, model, sess):
    res_list = []
    threaded_it = threaded_generator(valid_dset, 200)
    pred_list, label_list = [], []
    for batch in threaded_it:
        #res_dict, res_str, summary = model.run_batch(sess, batch, istrn=False)
        #res_list.append(res_dict)
        #plhs = [model.input_plh, model.mask_plh, model.label_plh, model.is_training]
        #feed = list(batch) + [True]
        #pred = sess.run(model.pred, dict(zip(plh, feed)))
        pred = model.classify(sess, batch[0], batch[1])
        #print(pred)
        pred_list.append(pred)
        label_list.append(np.asarray(batch[2]))
    pred = np.stack(pred_list)
    label = np.stack(label_list)
    return pred, label

def main():
    # load all args for the experiment
    args = utils.load_argparse_args(conf_dirs=conf_dirs)
    explogger = exp_logger.ExpLogger(args.log_prefix, args.save_dir, False)
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
    valid_dset = initDataset(args.valid_path, model.get_prepare_func(args), 1)
    test_dset = initDataset(args.test_path, model.get_prepare_func(args), 1)

    # step 3: Init tensorflow
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        if args.init_from:
            init_from = args.init_from
            for fn in os.listdir(args.init_from):
                print(fn)
                if fn.endswith('index'):
                    init_from = args.init_from + fn[:-6]
            model.saver.restore(sess, init_from)
            explogger.message('Model restored from {0}'.format(init_from))
        else:
            tf.global_variables_initializer().run()

        pred, label = validate(test_dset, model, sess)
        with open('res.pkl', 'wb') as f:
            pkl.dump(pred, f)
            pkl.dump(label, f)
            pkl.dump(args, f)
        print(pred, label)
        return pred, label, args
