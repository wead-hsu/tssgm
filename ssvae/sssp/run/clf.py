import sys
import os
import time
import pickle as pkl
from sssp.utils import utils
from sssp.config import exp_logger
from sssp.io.datasets import initDataset
from sssp.io.batch_iterator import threaded_generator
from sssp.utils.utils import average_res, res_to_string
import tensorflow as tf
import logging
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.DEBUG)

# ------------- CHANGE CONFIGURATIONS HERE ---------------
conf_dirs = ['sssp.config.conf_clf',]
#conf_dirs = ['sssp.config.conf_clf_multilabel']
# --------------------------------------------------------

def validate(valid_dset, model, sess, args, vocab, class_map):
    res_list = []
    threaded_it = threaded_generator(valid_dset, 200)
    wf = open(os.path.join(args.save_dir, 'gate.log'), 'w')
    rf = open(os.path.join(args.save_dir, 'predict.log'), 'w')
    rf.write('pred\ttarget\ttxt\n')
    list_predict = []
    list_target = []
    for batch in threaded_it:
        res_dict, res_str, summary, [gate_weights, pred] = model.run_batch(sess, batch, istrn=False)
        res_list.append(res_dict)
        
        batch_size = batch[0].shape[0]
        #wf.write(str(gate_weights))
        #wf.write(str(gate_weights.shape))
        for sidx in range(batch_size):
            for idx, w in enumerate(batch[0][sidx]):
                wf.write(vocab[w] + '\t')
                wf.write('{0:.3f}\t'.format(gate_weights[sidx][idx][0]) + '\n')
            wf.write('\n')

        pred = [class_map[idx] for idx in pred]
        tgt = [class_map[idx] for idx in batch[2]]
        list_predict.extend(pred)
        list_target.extend(tgt)
        inp = [[vocab[idx] for idx in s if idx != 0] for s in batch[0]]
        
        for i in range(batch_size):
            rf.write(str(pred[i]) + '\t')
            rf.write(str(tgt[i]) + '\t')
            rf.write(''.join(inp[i]) + '\n')

    wf.close()
    rf.close()
    
    cf = open(os.path.join(args.save_dir, 'confusion_matrix.log'), 'w')
    labels = sorted(list(class_map.values()))
    confusion = confusion_matrix(list_target, list_predict, labels)
    outstr = utils.confusion_matrix_to_string(confusion, labels)
    #print(outstr)
    cf.write(outstr)

    out_str = res_to_string(average_res(res_list))
    return out_str

def train_and_validate(args, model, sess, train_dset, valid_dset, test_dset, explogger, vocab, class_map):
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
            res_dict, res_str, summary, gate_weights = model.run_batch(sess, batch, istrn=True)
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
                out_str = validate(valid_dset, model, sess, args, vocab, class_map)
                explogger.message('VALIDATE: '  + out_str, True)

            if args.validate_every != -1 and batch_cnt % args.validate_every == 0:
                out_str = validate(test_dset, model, sess, args, vocab, class_map)
                explogger.message('TEST: ' + out_str, True)

            if args.save_every != -1 and batch_cnt % args.save_every == 0:
                save_fn = os.path.join(args.save_dir, args.log_prefix)
                explogger.message("Saving checkpoint model: {} ......".format(args.save_dir))
                model.saver.save(sess, save_fn, 
                        write_meta_graph=False,
                        global_step=batch_cnt)

            t_time = time.time()

def main():
    # load all args for the experiment
    args = utils.load_argparse_args(conf_dirs=conf_dirs)
    explogger = exp_logger.ExpLogger(args.log_prefix, args.save_dir)
    wargs = vars(args)
    wargs['conf_dirs'] = conf_dirs
    explogger.write_args(wargs)
    explogger.file_copy(['sssp'])

    # step 1: init dataset
    get_prepare_func = utils.get_prepare_clf_func_help(args)
    train_dset = initDataset(args.train_path, get_prepare_func(args), args.batch_size)
    valid_dset = initDataset(args.valid_path, get_prepare_func(args), args.batch_size)
    test_dset = initDataset(args.test_path, get_prepare_func(args), args.batch_size)
        
    if args.vocab_path:
        vocab = pkl.load(open(args.vocab_path, 'rb'))
        args.vocab_size = max(vocab.values())+1
        vocab = {int(vocab[k]): k for k in vocab}
    else:
        vocab = {i: 'NONE' for i in range(args.vocab_size)}
    if args.labels_path:
        class_map = pkl.load(open(args.labels_path, 'rb'))
        class_map = dict([[class_map[k], k.strip()] for k in class_map])
    else:
        class_map = dict([[k, str(k)] for k in range(args.num_classes)])

    # step 2: import specified model
    module = __import__(args.model_path, fromlist=[args.model_name])
    model_class = module.__dict__[args.model_name]
    model = model_class(args)
    vt, vs = model.model_setup(args)
    explogger.write_variables(vs)

    # step 3: Init tensorflow
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        if args.init_from:
            model.saver.restore(sess, args.init_from)
            explogger.message('Model restored from {0}'.format(args.init_from))
        else:
            tf.global_variables_initializer().run()

        train_and_validate(args, 
                model=model, 
                sess=sess, 
                train_dset=train_dset,
                valid_dset=valid_dset,
                test_dset=test_dset,
                explogger=explogger,
                vocab=vocab,
                class_map=class_map)
