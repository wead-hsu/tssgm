import sys
import os
import time
from sssp.utils import utils
from sssp.config import exp_logger
from sssp.io.datasets import initDataset
#from sssp.io.batch_iterator import threaded_generator
from sssp.utils.utils import average_res, res_to_string
import tensorflow as tf
import logging
import pickle as pkl

#logging.basicConfig(level=logging.INFO)

# ------------- CHANGE CONFIGURATIONS HERE ---------------
#conf_dirs = ['sssp.config.conf_semiclf_argparse',]
conf_dirs = ['sssp.config.conf_clf']
# --------------------------------------------------------

def validate(valid_dset, model, sess):
    res_list = []
    #threaded_it = threaded_generator(valid_dset, 200)
    for batch in valid_dset:
        res_dict, res_str, summary = model.run_batch(sess, batch, istrn=False)
        res_list.append(res_dict)
    out_str = res_to_string(average_res(res_list))
    return out_str

def get_batch(dataset):
    """ to get batch from an iterator, whenever the ending is reached. """
    while True:
        try:
            batch = dataset.next()
            break
        except:
            pass
    return batch

def run(args, model, sess, label_dset, unlabel_dset, valid_dset, test_dset, explogger):
    with open(args.vocab_path, 'rb') as f:
        vocab = pkl.load(f)
        vocab = {int(vocab[k]): k for k in vocab}

    batch_cnt = 0
    res_list = []

    # init tensorboard writer
    tb_writer = tf.summary.FileWriter(args.save_dir + '/train', sess.graph)

    t_time = time.time()
    time_cnt = 0
    for epoch_idx in range(args.max_epoch):
        #threaded_it_u = threaded_generator(unlabel_dset, 200)
        #threaded_it_l = threaded_generator(label_dset, 200)
        for batch_u in unlabel_dset:
            batch_cnt += 1
            batch_l = get_batch(label_dset)
            gen_time = time.time() - t_time
            t_time = time.time()
            res_dict, res_str, summary = model.run_batch(sess, batch_l+batch_u[:-1], istrn=True)
        
            plhs = [model.inp_l_plh,
                    model.tgt_l_plh,
                    model.msk_l_plh,
                    model.label_plh,
                    model.inp_u_plh,
                    model.tgt_u_plh,
                    model.msk_u_plh,]

            gates = sess.run(model.weights_l, dict(zip(plhs, batch_l+batch_u[:-1])))
            for sidx in range(20):
                for idx, w in enumerate(batch_l[1][sidx]):
                    print(vocab[w] + '\t', end='')
                #for idx, w in enumerate(batch_l[1][sidx]):
                    print('{0:.3f}\t'.format(gates[sidx][idx]))
                print()

            return

            run_time = time.time() - t_time
            res_dict.update({'run_time': run_time})
            res_dict.update({'gen_time': gen_time})
            res_list.append(res_dict)
            res_list = res_list[-200:]
            time_cnt += gen_time + run_time

            if batch_cnt % args.show_every == 0:
                tb_writer.add_summary(summary, batch_cnt)
                out_str = res_to_string(average_res(res_list))
                explogger.message(out_str)
            
            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = 'VALIDATE:' + validate(valid_dset, model, sess)
                explogger.message(out_str)

            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = 'TEST:' + validate(test_dset, model, sess)
                explogger.message(out_str)

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
    explogger = exp_logger.ExpLogger(args.log_prefix, 'results/vis')
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
    label_dset = initDataset(args.train_label_path, model.prepare_data, args.batch_size_label)
    unlabel_dset = initDataset(args.train_unlabel_path, model.prepare_data, args.batch_size_unlabel)
    valid_dset = initDataset(args.valid_path, model.prepare_data, args.batch_size_label)
    test_dset = initDataset(args.test_path, model.prepare_data, args.batch_size_label)

    # step 3: Init tensorflow
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        #init_from = 'results/zhongao/taggatedgru-hasnan/Code_Zasd_ybsd/rnn_test-4000'
        model.saver.restore(sess, args.init_from)
        explogger.message('Model restored from {0}'.format(init_from))

        run(args, 
                model=model, 
                sess=sess, 
                label_dset=label_dset,
                unlabel_dset=unlabel_dset,
                valid_dset=valid_dset,
                test_dset=test_dset,
                explogger=explogger)
