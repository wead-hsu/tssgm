from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.utils import smart_cond
from tensorflow.core.protobuf import saver_pb2
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.contrib.bayesflow import stochastic_graph as sg
from tensorflow.contrib.bayesflow import stochastic_gradient_estimators as sge
#from tensorflow.python.framework import ops

import logging
import numpy as np
import pickle as pkl
from sssp.models.model_base import ModelBase
from sssp.models.layers.classifier import *
from sssp.utils.utils import res_to_string

logging.basicConfig(level=logging.INFO)

class SemiClassifier(ModelBase):
    def __init__(self, args):
        super(SemiClassifier, self).__init__()
        self._logger = logging.getLogger(__name__)

    def _get_rnn_cell(self, rnn_type, num_units, num_layers):
        if rnn_type == 'LSTM':
            # use concated state for convinience
            cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
        elif rnn_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units)
        else:
            raise 'The rnn type is not supported.'

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        return cell

    def _create_placeholders(self):
        self.inp_l_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='inp_l_plh')

        self.tgt_l_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='tgt_l_plh')

        self.msk_l_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='msk_l_plh')

        self.inp_u_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='inp_u_plh')

        self.tgt_u_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='tgt_u_plh')

        self.msk_u_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='msk_u_plh')

        self.is_training_plh = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='is_training_plh')

        self.keep_prob_plh = tf.placeholder(
                dtype=tf.float32,
                shape=[], 
                name='keep_prob_plh')

        self.beam_size_plh = tf.placeholder(
                tf.int32,
                shape=[],
                name='beam_size_plh')

        self.label_plh = tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='label')

    def _create_embedding_matrix(self, args):
        if args.embd_path is None:
            np.random.seed(1234567890)
            embd_init = np.random.randn(args.vocab_size, args.embd_dim).astype(np.float32) * 1e-3
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

    def _create_encoder(self, inp, seqlen, scope_name, args):
        with tf.variable_scope(scope_name):
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            emb_inp = tf.nn.dropout(emb_inp, self.keep_prob_plh)

            cell = self._get_rnn_cell(args.rnn_type, args.num_units, args.num_layers)
            _, enc_state = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=emb_inp,
                    dtype=tf.float32,
                    sequence_length=seqlen)

            if args.rnn_type == 'LSTM':
                enc_state = enc_state[-1]

            return enc_state

    def _create_decoder(self, inp, mask, init_state, label_oh, scope_name, args):
        with tf.variable_scope(scope_name):

            if args.decoder_type == 'sclstm':
                self._logger.info('Using ScLSTM as the decoder')
                from sssp.models.layers.sclstm import ScLSTM
                emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
                emb_inp = tf.nn.dropout(emb_inp, self.keep_prob_plh)
                y_inp = tf.tile(label_oh[:, None, :], [1, tf.shape(emb_inp)[1], 1])
                sclstm_layer = ScLSTM(args.embd_dim, args.num_units, args.num_classes)
                _, dec_outs = sclstm_layer.forward(emb_inp, mask, y_inp, return_final=False, initial_state=(init_state, init_state))
                cell = sclstm_layer._lstm_step
            else:
                self._logger.info('Using LSTM as the decoder')
                emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
                emb_inp = tf.concat([emb_inp, tf.tile(label_oh[:, None, :], [1, tf.shape(emb_inp)[1], 1])], axis=2)
                emb_inp = tf.nn.dropout(emb_inp, self.keep_prob_plh)

                cell = self._get_rnn_cell(args.rnn_type, args.num_units, args.num_layers)

                dec_outs, _ = tf.nn.dynamic_rnn(
                        cell=cell,
                        inputs=emb_inp,
                        sequence_length=tf.to_int32(tf.reduce_sum(mask, axis=1)),
                        initial_state=tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
                        )

            w_t = tf.get_variable(
                    'proj_w', 
                    [args.vocab_size, args.num_units])
            b = tf.get_variable(
                    "proj_b", 
                    [args.vocab_size])
            out_proj = (w_t, b)

        def dec_step_func(emb_t, hidden_state):
            with tf.variable_scope(scope_name):
                with tf.variable_scope('rnn', reuse=True):
                    out, state = cell(emb_t, hidden_state)
            logits = tf.add(tf.matmul(out, w_t, transpose_b=True), b[None, :])
            prob = tf.nn.log_softmax(logits)
            return state, prob

        return dec_outs, out_proj, dec_step_func, cell

    def _create_rnn_classifier(self, inp, msk, keep_rate, scope_name, args):
        enc_state, weights = create_encoder(self,
                inp=inp,
                msk=msk,
                keep_rate=args.keep_rate,
                scope_name=scope_name,
                args=args)

        logits = create_fclayers(self,
                enc_state=enc_state,
                num_classes=args.num_classes,
                args=args,
                scope='fc')

        return logits, weights

    def _create_softmax_layer(self, proj, dec_outs, targets, weights, scope_name, args):
        with tf.variable_scope(scope_name):
            w_t, b = proj

            # is_training = Flase
            def get_llh_test():
                dec_outs_flt = tf.reshape(dec_outs, [-1, args.num_units])
                logits_flt = tf.add(tf.matmul(dec_outs_flt, w_t, transpose_b=True), b[None, :])
                logits = tf.reshape(logits_flt, [tf.shape(dec_outs)[0], -1, args.vocab_size])
    
                llh_precise = tf.contrib.seq2seq.sequence_loss(
                        logits=logits,
                        targets=targets,
                        weights=weights,
                        average_across_timesteps=False,
                        average_across_batch=False,
                        softmax_loss_function=None)
                return llh_precise
            
            # is_training = True
            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # use 32bit float to avoid numerical instabilites
                #w_t = tf.transpose(w)
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_softmax_loss(
                        weights=local_w_t, 
                        biases=local_b,
                        inputs=local_inputs, 
                        labels=labels, 
                        num_sampled=args.num_samples,
                        num_classes=args.vocab_size,
                        partition_strategy="div")
                
            # is_training = False
            def get_llh_train():
                # if use sampled_softmax
                if args.use_sampled_softmax and args.num_samples > 0 and args.num_samples < args.vocab_size:
                    llh_train = tf.contrib.seq2seq.sequence_loss(
                            logits=dec_outs,
                            targets=targets,
                            weights=weights,
                            average_across_timesteps=False,
                            average_across_batch=False,
                            softmax_loss_function=sampled_loss)
                    self._logger.info('Use sampled softmax during training')
                else:
                    llh_train = get_llh_test()
                    self._logger.info('Use precise softmax during training')
                return llh_train
    
            loss = smart_cond(self.is_training_plh, get_llh_train, get_llh_test)
        return loss

    def _get_elbo_label(self, inp, tgt, msk, label, args):
        """ Build encoder and decoders """
        xlen = tf.to_int32(tf.reduce_sum(msk, axis=1))
        enc_state = self._create_encoder(
                tgt,
                seqlen=xlen,
                scope_name='enc',
                args=args)
        enc_state = tf.nn.dropout(enc_state, self.keep_prob_plh)
        enc_state = tf.contrib.layers.fully_connected(enc_state, 
                num_outputs=args.num_units,
                activation_fn=None,
                scope='x_to_a')
        enc_state = tf.contrib.layers.batch_norm(enc_state,
                center=True,
                scale=True,
                is_training=self.is_training_plh,
                scope='bn_a')
        enc_state = tf.tanh(enc_state)
        enc_state = tf.nn.dropout(enc_state, self.keep_prob_plh)

        label_oh = tf.gather(tf.eye(args.num_classes), label)
        with tf.variable_scope('latent'):
            y_enc_in = tf.contrib.layers.fully_connected(label_oh, args.dim_z, scope='y_enc_in')
            y_enc_in = tf.nn.dropout(y_enc_in, self.keep_prob_plh)
            pst_in = tf.concat([y_enc_in, enc_state], axis=1)
            pst_in = tf.contrib.layers.fully_connected(pst_in, args.num_units, None, scope='pst_in_dense')
            pst_in = tf.contrib.layers.batch_norm(pst_in, center=True, scale=True, 
                    is_training=self.is_training_plh, scope='pst_in_bn')
            pst_in = tf.tanh(pst_in)
            pst_in = tf.nn.dropout(pst_in, self.keep_prob_plh)
            mu_pst = tf.contrib.layers.fully_connected(pst_in, args.dim_z, tf.nn.tanh,
                    scope='mu_posterior')
            logvar_pst = tf.contrib.layers.fully_connected(pst_in, args.dim_z, tf.nn.tanh,
                    scope='logvar_posterior')
            mu_pri = tf.zeros_like(mu_pst)
            logvar_pri = tf.ones_like(logvar_pst)
            dist_pri = tf.contrib.distributions.Normal(mu=mu_pri, sigma=tf.exp(logvar_pri))
            dist_pst = tf.contrib.distributions.Normal(mu=mu_pst, sigma=tf.exp(logvar_pst))
            kl_loss = tf.contrib.distributions.kl(dist_pst, dist_pri)
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

        with st.value_type(st.SampleValue(stop_gradient=False)):
            z_st_pri = st.StochasticTensor(dist_pri, name='z_pri')
            z_st_pst = st.StochasticTensor(dist_pst, name='z_pst')
            z = smart_cond(self.is_training_plh, lambda: z_st_pst, lambda: z_st_pri)
       
        z_ext = tf.contrib.layers.fully_connected(tf.reshape(z, [-1, args.dim_z]), args.num_units, scope='extend_z')
        z_ext = tf.nn.dropout(z_ext, self.keep_prob_plh)
        yz = tf.concat([z_ext, label_oh], axis=1)
        yz = tf.contrib.layers.fully_connected(yz, args.num_units, None, scope='yz_dense')
        yz = tf.contrib.layers.batch_norm(yz, center=True, scale=True, is_training=self.is_training_plh, scope='yz_bn')
        yz = tf.tanh(yz)
        yz = tf.nn.dropout(yz, self.keep_prob_plh)
        xlen = tf.to_int32(tf.reduce_sum(msk, axis=1))
        outs, proj, dec_func, cell  = self._create_decoder(
                inp,
                mask=msk,
                label_oh=label_oh,
                init_state=yz, #tf.contrib.rnn.LSTMStateTuple(yz, yz),
                scope_name='dec',
                args=args)
        outs = tf.nn.dropout(outs, self.keep_prob_plh)

        # build loss layers
        recons_loss = self._create_softmax_layer(
                proj=proj,
                dec_outs=outs,
                targets=tgt,
                weights=msk,
                scope_name='loss',
                args=args)
        recons_loss = recons_loss * msk
        
        return recons_loss, kl_loss
    
    def get_loss_l(self, args):
        with tf.variable_scope(args.log_prefix):
            """ label CVAE """
            self.recons_loss_l, self.kl_loss_l = self._get_elbo_label(self.inp_l_plh,
                    self.tgt_l_plh,
                    self.msk_l_plh,
                    self.label_plh,
                    args)
            self.recons_loss_l = self.recons_loss_l * self.msk_l_plh
            self.ppl_l = tf.exp(tf.reduce_sum(self.recons_loss_l) / tf.reduce_sum(self.msk_l_plh))
            self.recons_loss_l = tf.reduce_sum(self.recons_loss_l * self.msk_l_plh, axis=1)
            self.recons_loss_l = tf.reduce_mean(self.recons_loss_l)
            self.kl_loss_l = tf.reduce_mean(self.kl_loss_l)
            self.elbo_loss_l = self.recons_loss_l + self.kl_loss_l * self.kl_w
            
            """ label CLASSIFICATION """
            self.logits_l, weights_l = self._create_rnn_classifier(self.tgt_l_plh,
                    self.msk_l_plh,
                    keep_rate=args.keep_rate,
                    scope_name='clf',
                    args=args)
            self.predict_loss_l = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label_plh,
                    logits=self.logits_l)
            self.predict_loss_l = tf.reduce_mean(self.predict_loss_l)
            self.accuracy_l = tf.equal(tf.cast(self.label_plh, tf.int64), tf.argmax(self.logits_l, axis=1))
            self.accuracy_l = tf.reduce_mean(tf.cast(self.accuracy_l, tf.float32))
            self.loss_l = self.elbo_loss_l + self.predict_loss_l * args.alpha

            tf.summary.scalar('elbo_loss_l', self.elbo_loss_l)
            tf.summary.scalar('kl_w', self.kl_w)
            tf.summary.scalar('ppl_l', self.ppl_l)
            tf.summary.scalar('kl_loss_l', self.kl_loss_l)
            tf.summary.scalar('pred_loss_l', self.predict_loss_l)
            tf.summary.scalar('accuracy_l', self.accuracy_l)
        return self.loss_l
    
    def get_loss_u_sample(self, args):
        with tf.variable_scope(args.log_prefix, reuse=True):
            """ unlabel CLASSIFICATION """
            self.logits_u, weights_u = self._create_rnn_classifier(self.tgt_u_plh,
                    self.msk_u_plh,
                    keep_rate=args.keep_rate,
                    scope_name='clf',
                    args=args)
            self.predict_u = tf.nn.softmax(self.logits_u)

            """ unlabel CVAE """
            with st.value_type(st.SampleValue(stop_gradient=True)):
                y_st = st.StochasticTensor(tf.contrib.distributions.Categorical(p=self.predict_u), 
                        name='dist_y',
                        loss_fn=sge.get_score_function_with_baseline())
                recons_loss_u_s, kl_loss_u_s = self._get_elbo_label(self.inp_u_plh,
                        self.tgt_u_plh,
                        self.msk_u_plh,
                        y_st,
                        args)
                recons_loss_u_s = recons_loss_u_s  * self.msk_u_plh

                # routing_loss for sampling-based classifier
                if args.use_weights:
                    self._logger.info('Use Reweighting Approach')
                    if args.use_binaryweights:
                        self._logger.info('Use Binary Reweighting Approach')
                        weights_u = tf.cast(tf.greater(weights_u, 1/tf.reduce_sum(self.msk_u_plh, axis=1)[:, None, None]), tf.float32)
                    routing_loss = tf.reduce_sum(recons_loss_u_s*self.msk_u_plh*weights_u[:,:,0], axis=1)
                    routing_loss = routing_loss / tf.reduce_sum(weights_u[:,:,0], axis=1)
                else:
                    routing_loss = tf.reduce_sum(recons_loss_u_s*self.msk_u_plh, axis=1)
                    routing_loss = routing_loss / tf.reduce_sum(self.msk_u_plh, axis=1)
                routing_loss += kl_loss_u_s * self.kl_w
                
                # loss for the generator/decoder
                loss_u_of_gen = tf.reduce_sum(recons_loss_u_s, axis=1) 
                loss_u_of_gen += kl_loss_u_s * self.kl_w

        with tf.variable_scope(args.log_prefix, reuse=False):
            surrogate_loss = y_st.loss(routing_loss)
            self.entropy_u = tf.losses.softmax_cross_entropy(self.predict_u, self.predict_u)

        return tf.reduce_mean(surrogate_loss) + tf.reduce_mean(loss_u_of_gen) + self.entropy_u
    
    def get_loss_u(self, args):
        self._logger.info('Reweighting approach is not valid without sampling')
        with tf.variable_scope(args.log_prefix, reuse=True):
            """ unlabel CVAE """
            self.recons_loss_u, self.kl_loss_u, self.loss_sum_u = [], [], []
            for idx in range(args.num_classes):
                label_i = tf.constant(idx)
                label_i = tf.tile([idx], [tf.shape(self.tgt_u_plh)[0]])
                recons_loss_ui, kl_loss_ui = self._get_elbo_label(self.inp_u_plh,
                        self.tgt_u_plh,
                        self.msk_u_plh,
                        label_i, 
                        args)
                self.recons_loss_u.append(tf.reduce_sum(recons_loss_ui, axis=1))
                self.kl_loss_u.append(kl_loss_ui)
                self.loss_sum_u.append(tf.reduce_sum(recons_loss_ui, axis=1) + kl_loss_ui * self.kl_w)
            
            """ unlabel CLASSIFICATION """
            self.logits_u, self.weights_u = self._create_rnn_classifier(self.tgt_u_plh,
                    self.msk_u_plh,
                    keep_rate=args.keep_rate,
                    scope_name='clf',
                    args=args)
            self.predict_u = tf.nn.softmax(self.logits_u)
            self.entropy_u = tf.losses.softmax_cross_entropy(self.predict_u, self.predict_u)

            self.loss_sum_u = tf.add_n([self.loss_sum_u[idx] * self.predict_u[:, idx] for idx in range(args.num_classes)]) # [bs]
            self.loss_sum_u = tf.reduce_mean(self.loss_sum_u)
        return self.loss_sum_u + self.entropy_u

    def model_setup(self, args):
        with tf.variable_scope(args.log_prefix):
            self.init_global_step()
            self._create_placeholders()
            self._logger.info("Created placeholders.")
            self._create_embedding_matrix(args)

            self.kl_w = tf.log(1. + tf.exp((self.global_step - args.klw_b) * args.klw_w))
            self.kl_w = tf.minimum(self.kl_w, 1.) / 100.0 #scale reweighted
        
        self.loss_l = self.get_loss_l(args)
        self.train_unlabel = tf.greater(self.global_step, args.num_pretrain_steps)
        if args.sample_unlabel == 'False':
            self.loss_u = smart_cond(self.train_unlabel, lambda: self.get_loss_u(args), lambda: tf.constant(0.))
        elif args.sample_unlabel == 'S1':
            self.loss_u = smart_cond(self.train_unlabel, lambda: self.get_loss_u_sample(args), lambda: tf.constant(0.))
        else:
            raise Exception('sampling method is not supported. {}'.format(args.sample_unlabel))
        tf.summary.scalar('train_unlabel', tf.to_int64(self.train_unlabel))
        tf.summary.scalar('loss_u', self.loss_u)

        self.loss = self.loss_l + self.loss_u
        tf.summary.scalar('loss', self.loss)

        with tf.variable_scope(args.log_prefix):
            # optimizer
            #embd_var = self.embedding_matrix
            #other_var_list = [v for v in tf.trainable_variables() if v.name != embd_var.name]
            #learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                    #args.decay_steps,
                    #args.decay_rate,
                    #staircase=True)
            self.train_op = self.training_op(self.loss, #tf.trainable_variables(),
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created SemiClassifier Model.")

            self._create_saver(args)
            self._logger.info('Created Saver')

            self.merged = tf.summary.merge_all()

            """ Create beam search layer
            self.beam_output_cur, self.beam_scores_cur = self._create_beam_search_layer(
                    init_state=yz,
                    dec_step_func=cur_dec_func,
                    cell=cur_cell,
                    embedding_matrix=self.embedding_matrix,
                    vocab_size=args.vocab_size,
                    num_layers=args.num_layers,)
            self._logger.info('Created Beam Search Layer')
            """

            vt = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix)
            vs = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=args.log_prefix)
        return vt, vs

    def run_batch(self, sess, inps, istrn, args):
        plhs = [self.inp_l_plh,
                self.tgt_l_plh,
                self.msk_l_plh,
                self.label_plh,
                self.inp_u_plh,
                self.tgt_u_plh,
                self.msk_u_plh,]

        if istrn:
            fetch_dict = [['elbo_l', self.elbo_loss_l],
                    ['ppl_l', self.ppl_l],
                    ['kl_l', self.kl_loss_l],
                    ['pred_l', self.predict_loss_l],
                    ['acc_l', self.accuracy_l],
                    ['loss_u', self.loss_u],
                    ['train_u', self.train_unlabel],
                    ['klw', self.kl_w]]

            feed_dict = dict(list(zip(plhs, inps)) + [[self.is_training_plh, True], [self.keep_prob_plh, args.keep_rate]])
            fetch = [self.merged] + [t[1] for t in fetch_dict] + [self.train_op]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        else:
            fetch_dict = [['pred_l', self.predict_loss_l],
                    ['acc_l', self.accuracy_l],]
            feed_dict = dict(list(zip(plhs, inps+inps[:-1])) + [[self.is_training_plh, False], [self.keep_prob_plh, 1.0]])
            fetch = [self.merged] + [t[1] for t in fetch_dict]
            res = sess.run(fetch, feed_dict)
            res_dict = dict([[fetch_dict[i][0], res[i+1]] for i in range(len(fetch_dict))])
            res_str = res_to_string(res_dict)
        return res_dict, res_str, res[0]

    def _create_saver(self, args):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.VARIABLES, scope=args.log_prefix),
                max_to_keep=args.max_to_keep,
                write_version=saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver
    
    def get_prepare_func(self, args):
        def prepare_data(raw_inp):
            raw_inp = [[s.split(' ') for s in l.strip().split('\t')] for l in raw_inp[0]]
            raw_inp = list(zip(*raw_inp))
            label, inp = raw_inp
            
            def proc(sents):
                sent_lens = [len(s) for s in sents]
                max_sent_len = min(args.max_sent_len, max(sent_lens))
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
            label = np.asarray(label).flatten().astype('int32')
            #print(inp + (label,))
            return inp + (label,)
        return prepare_data

