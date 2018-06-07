from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.utils import smart_cond
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import array_ops

import logging
import numpy as np
import pickle as pkl
from sssp.models.model_base import ModelBase
from sssp.utils.utils import res_to_string
from sssp.models.layers.gru import GRU
from sssp.models.layers.lstm import LSTM
from sssp.models.layers.gated_lstm import GatedLSTM
from sssp.models.layers.classifier import create_encoder, create_fclayers

class RnnClassifier(ModelBase):
    def __init__(self, args):
        super(RnnClassifier, self).__init__()
        self._logger = logging.getLogger(__name__)
       
    def _create_placeholders(self, args):
        self.input_plh = tf.placeholder(
                dtype=tf.int64,
                shape=[None, None if args.fix_sent_len <=0 else args.fix_sent_len],
                name='input_plh')

        self.mask_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='mask_plh')

        self.label_plh = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='label_plh')

        self.is_training_plh = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='is_training_plh')

    def _create_embedding_matrix(self, args):
        if args.embd_path is None:
            np.random.seed(1234567890)
            embd_init = np.random.randn(args.vocab_size, args.embd_dim).astype(np.float32) * 1e-2
        else:
            embd_init = pkl.load(open(args.embd_path, 'rb'))
            assert embd_init.shape[0] == args.vocab_size \
                    and embd_init.shape[1] == args.embd_dim, \
                'Shapes between given pretrained embedding matrix and given settings do not match'

        with tf.variable_scope('embedding_matrix'):
            self.embedding_matrix = tf.get_variable(
                    'embedding_matrix', 
                    [args.vocab_size, args.embd_dim],
                    initializer=tf.constant_initializer(embd_init))

    def model_setup(self, args):
        with tf.variable_scope(args.log_prefix):
            self.init_global_step()
            self._create_placeholders(args)
            self._create_embedding_matrix(args)

            batch_size = tf.shape(self.input_plh)[0]
            seqlen = tf.to_int64(tf.reduce_sum(self.mask_plh, axis=1))
            enc_state, gate_weights = create_encoder(self,
                    inp=self.input_plh,
                    msk=self.mask_plh,
                    keep_rate=args.keep_rate,
                    scope_name='enc_rnn',
                    args=args)
            self.gate_weights = gate_weights

            self.loss_reg_l1 = tf.reduce_mean(tf.reduce_sum(gate_weights, axis=1))
            self.loss_reg_diff = tf.reduce_sum(tf.abs(gate_weights[:, :-1, :] - gate_weights[:, 1:, :]), axis=1)
            self.loss_reg_diff = tf.reduce_mean(self.loss_reg_diff)
            self.loss_reg_sharp = tf.reduce_sum(gate_weights * (1-gate_weights), axis=1)
            self.loss_reg_sharp = tf.reduce_mean(self.loss_reg_sharp)
            self.loss_reg_frobenius =  - tf.reduce_mean(tf.reduce_sum(gate_weights*gate_weights, axis=1))

            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', type(args.fixirrelevant))

            if args.fixirrelevant == False:
                logits = create_fclayers(self, enc_state, args.num_classes, 'fclayers', args)
                self.prob = tf.nn.softmax(logits)
                self.pred = tf.argmax(logits, axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_plh), tf.float32))
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_plh, logits=logits)
                self.loss = tf.reduce_mean(self.loss)
            else:
                print('Use fixirrelevant')
                logits = create_fclayers(self, enc_state, args.num_classes-1, 'fclayers', args)
                full_logits = tf.concat([tf.zeros([batch_size,1]), logits], axis=1) # the irrelevant is concatenated on the left
                self.prob = tf.nn.softmax(full_logits)
                self.pred = tf.argmax(full_logits, axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_plh), tf.float32))
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_plh, logits=full_logits)
                self.loss = tf.reduce_mean(self.loss)
            
            self.loss_total = self.loss + args.w_regl1 * self.loss_reg_l1 + \
                    args.w_regdiff * self.loss_reg_diff + \
                    args.w_regsharp * self.loss_reg_sharp  + \
                    args.w_regfrobenius * self.loss_reg_frobenius 
            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
            self.train_op = self.training_op(self.loss_total, 
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created RnnClassifier.")
            self._create_saver(args)
            self._logger.info('Created Saver')

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reg_l1', self.loss_reg_l1)
            tf.summary.scalar('reg_diff', self.loss_reg_diff)
            tf.summary.scalar('reg_sharp', self.loss_reg_sharp)
            tf.summary.scalar('reg_frobenius', self.loss_reg_sharp)
            tf.summary.scalar('loss_total', self.loss_total)
            self.merged = tf.summary.merge_all()

            vt = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix)
            vs = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix)
        return vt, vs

    def run_batch(self, sess, inps, istrn=True):
        plhs = [self.input_plh,
                self.mask_plh,
                self.label_plh]

        if istrn:
            fetch_dict = [
                    ['loss_total', self.loss_total],
                    ['loss', self.loss],
                    ['reg_l1', self.loss_reg_l1],
                    ['reg_diff', self.loss_reg_diff],
                    ['reg_sharp', self.loss_reg_sharp],
                    ['reg_frobenius', self.loss_reg_frobenius],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training_plh, istrn]])
            fetch_nonscalar = [self.merged, self.gate_weights, self.pred]
            fetch = fetch_nonscalar + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+len(fetch_nonscalar)]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        else:
            fetch_dict = [
                    ['loss_total', self.loss_total],
                    ['loss', self.loss],
                    ['loss_reg_l1', self.loss_reg_l1],
                    ['loss_reg_diff', self.loss_reg_diff],
                    ['loss_reg_sharp', self.loss_reg_sharp],
                    ['acc', self.accuracy],
                    ]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training_plh, istrn]])
            fetch_nonscalar = [self.merged, self.gate_weights, self.pred]
            fetch = fetch_nonscalar + [t[1] for t in fetch_dict]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+len(fetch_nonscalar)]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        return res_dict, res_str, res[0], res[1: len(fetch_nonscalar)]
    
    def classify(self, sess, sent, mask):
        feed_dict = {self.input_plh: sent, self.mask_plh: mask, self.is_training_plh: False}
        fetch = [self.prob]
        prob = sess.run(fetch, feed_dict)
        return prob

    def _create_saver(self, args):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix),
                max_to_keep=args.max_to_keep,
                write_version=saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver

    def get_prepare_func(self, args):
        def prepare_data(inp):
            inp = [[s.split(' ') for s in l.strip().split('\t')] for l in inp[0]]
            inp = list(zip(*inp))
            label, inp = inp
             
            def proc(sents):
                sent_lens = [len(s) for s in sents]
                max_sent_len = min(args.max_sent_len, max(sent_lens))
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
            inp = (inp[1], inp[2])
            label = np.asarray(label).flatten().astype('int64')

            return inp + (label,)
        return prepare_data
