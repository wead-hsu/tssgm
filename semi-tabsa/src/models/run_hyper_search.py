
import sys
import csv
import os
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperparam_config import params
import tensorflow as tf 
import semi_tabsa_bilstm_att as tssvae_bilstm_att_g
import semi_tabsa_ian as tssvae_ian
from imp import reload
global trial_counter
global log_handler


def hyperopt_wrapper(param):
    global trial_counter
    global log_handler
    trial_counter += 1

    acc, f1 = hyperopt_obj(param, trial_counter)
    hyperparam_to_log = [ 
        "%d" % trial_counter,
        "%.4f"% acc,
        "%.4f"% f1
    ]

    for k,v in param.items():
        hyperparam_to_log.append(v)
    writer.writerow(hyperparam_to_log)
    log_handler.flush()
   
    return {'loss': acc, 'attchments':{'f1': f1}, 'status': STATUS_OK}

def hyperopt_obj(param, trial_counter):
    FLAGS=tf.app.flags.FLAGS
    FLAGS.batch_size = int(param['batch_size'])
    FLAGS.n_hidden = int(param['n_hidden'])
    FLAGS.n_hidden_ae = int(param['n_hidden_ae'])
    FLAGS.learning_rate = param['learning_rate']
    FLAGS.n_unlabel = int(param['n_unlabel'])
    FLAGS.l2_reg = param['l2_reg']
    FLAGS.n_iter = int(param['n_iter'])
    FLAGS.keep_rate = param['keep_rate']
    FLAGS.decoder_type = param['decoder_type']
    FLAGS.grad_clip = param['grad_clip']
    FLAGS.dim_z = int(param['dim_z'])
    FLAGS.alpha = param['alpha']
    FLAGS.position_enc = param['position_enc']
    FLAGS.bidirection_enc = param['bidirection_enc']
    FLAGS.bidirection_dec = param['bidirection_dec']
    FLAGS.position_dec = param['position_dec']
    if param['task'] == "BILSTM_ATT_G":
        FLAGS.classifier_type = "BILSTM_ATT_G"
    elif param['task'] == "IAN":
        FLAGS.classifier_type = "IAN"
    FLAGS.data_dir='unlabel_lapt_10k'
  #  FLAGS.n_unlabel = 100
  #  FLAGS.batch_size = 64
  #  FLAGS.n_iter = 1
    tf.reset_default_graph()
    acc, f1 = tssvae_bilstm_att_g.main(FLAGS) 
    
    return acc, f1
    

if __name__ == "__main__":
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
    tf.app.flags.DEFINE_integer('n_hidden_ae', 200, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 95, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
    tf.app.flags.DEFINE_integer('n_unlabel', 10000, 'number of unlabeled')
    
    tf.app.flags.DEFINE_string('train_file_path', '../../../data/se2014task06/tabsa-lapt/train.pkl', 'training file')
    tf.app.flags.DEFINE_string('unlabel_file_path', '../../../data/se2014task06/tabsa-lapt/unlabel.clean.pkl', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', '../../../data/se2014task06/tabsa-lapt/dev.pkl', 'training file')
    tf.app.flags.DEFINE_string('test_file_path', '../../../data/se2014task06/tabsa-lapt/test.pkl', 'training file')
    tf.app.flags.DEFINE_string('classifier_type', 'BILSTM_ATT_G', 'model type: ''(default), MEM or TD or TC or BILSTM_ATT_G')
    tf.app.flags.DEFINE_float('keep_rate', 0.25, 'keep rate')
    tf.app.flags.DEFINE_string('decoder_type', 'lstm', '[sclstm, lstm]')
    tf.app.flags.DEFINE_float('grad_clip', 100, 'gradient_clip, <0 == None')
    tf.app.flags.DEFINE_integer('dim_z', 50, 'dimension of z latent variable')
    tf.app.flags.DEFINE_float('alpha', 5.0, 'weight of alpha')
    tf.app.flags.DEFINE_string('save_dir', '.', 'directory of save file')
    tf.app.flags.DEFINE_string('position_enc', '', '[binary, distance, ], None for no position embedding')
    tf.app.flags.DEFINE_boolean('bidirection_enc', False, 'boolean')
    tf.app.flags.DEFINE_string('position_dec', '', '[binary, distance, ], None for no position embedding')
    tf.app.flags.DEFINE_boolean('bidirection_dec', False , 'boolean')
    
    specified_models = sys.argv[1] 
    param = params[specified_models]
    headers = ['trial_counter', 'acc', 'f1_marco']
    log_file = "./%s_hyperopt.log" %(specified_models)
    log_handler = open( log_file, "w")
    writer = csv.writer(log_handler)

    for k,v in sorted(param.items()):
        headers.append(k)
    writer.writerow(headers)
    log_handler.flush()

    trial_counter = 0
    trials = Trials()
    objective = lambda p:hyperopt_wrapper(p)
    best_params = fmin(objective, param, algo=tpe.suggest, trials=trials, max_evals=300)

    for k,v in best_params.items():
        print("    %s: %s" %(k,v))
    
