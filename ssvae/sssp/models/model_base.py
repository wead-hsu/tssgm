import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import pickle
#from sssp.models.async_optimizer import AsyncAdamOptimizer

class ModelBase(object):
    def __init__(self):
        pass
    
    def model_setup(self, args):
        raise NotImplementedError

    def run_batch(self, sess, inp, istrn=True):
        raise NotImplementedError

    def init_global_step(self):
        # Global steps for asynchronous distributed training.
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)
    
    def async_training_op(self, cost, embd_var, other_var_list,
            grad_clip=5.0,
            max_norm=200.0,
            learning_rate=0.01,
            grads=None, 
            train_embd=True):
        '''
        2016-11-15, Haoze Sun
        0. gradient for word embeddings is a tf.sparse_tensor
        1. clip_norm and clip by value operations do not support sparse tensor
        2. When using Adam, it seems word embedding is not trained on GPU.
            print(embd_var.name)
           Actually, CPU is not capable to execute 8-worker word embedding Adam updating, which
            cause the GPU usage-->0% and the train is very slow.
        3. We employ AdaGrad instead, if args.train_embd == True.
           Gradient clip is barely used in AdaGrad.
           Other optimizator like RMSProp, Momentum have not tested.

        ref: http://stackoverflow.com/questions/40621240/gpu-cpu-tensorflow-training
        ref: http://stackoverflow.com/questions/36498127/
                  how-to-effectively-apply-gradient-clipping-in-tensor-flow
        ref: http://stackoverflow.com/questions/35828037/
                  training-a-cnn-with-pre-trained-word-embeddings-is-very-slow-tensorflow

        To Solve the problem:
        ref: https://github.com/tensorflow/tensorflow/issues/6460
        '''

        # ------------- calc gradients --------------------------
        var_list = [embd_var] + other_var_list
        if grads is None:
            grads = tf.gradients(cost, var_list)

        # ------------- Optimization -----------------------
        # 0. global step used for asynchronous distributed training.
        # 1. Adam (default lr 0.0004 for 8 GPUs, 300 batchsize) if args.train_embd == False,
        #    apply gradient clip operations (default 10, 100)
        # 2. Adagrad (default 0.01~0.03? 1e-8? for 8 GPUs, 300 batchsize) if train embedding,
        #    no gradient clip.
        #    However, Adagrad is not suitable for large datasets.
        # 3. Momentum (default 0.001)
        # 4. RMSProp/Adadelta (default 0.001) is also OK......

        if not train_embd:
            # -------------------- clip all gradients--------------
            grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, max_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            word_grad, other_grads = [grads[0]], grads[1:]
            # -------------------- clip all gradients except word embedding gradient --------------
            other_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in other_grads]
            other_grads, _ = tf.clip_by_global_norm(other_grads, max_norm)
            grads = word_grad + other_grads  # keep the order
            # optimizer = tf.train.AdamOptimizer(args.lr)
            # optimizer = tf.train.RMSPropOptimizer(args.lr)
            optimizer = AsyncAdamOptimizer(learning_rate)

        if not hasattr(self, 'global_step'):
            self.init_global_step()

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def training_op(self, cost, var_list,
            grad_clip=-1,
            max_norm=-1,
            learning_rate=0.001,
            grads=None, 
            train_embd=True):
        
        # ------------- calc gradients --------------------------
        if grads is None:
            grads = tf.gradients(cost, var_list)

        for i, g in enumerate(grads):
            if g is None:
                print('WARNING: {} is not in the graph'.format(var_list[i].name))

        if grad_clip > 0: grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in grads]
        if max_norm > 0: grads, _ = tf.clip_by_global_norm(grads, max_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        if not hasattr(self, 'global_step'):
            self.init_global_step()

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def _create_beam_search_layer(self, init_state, dec_step_func, embedding_matrix, vocab_size,
            cell, SOS_ID=0, EOS_ID=0, num_layers=1):
        """ Tensorflow implementation for beam searching, source:
        https://github.com/sdlg/nlc/blob/master/nlc_model.py
        Many thanks.

        Args:
            init_state: initial state used in decoder, batch_size should be 1.
            dec_step_func: (inp, state) -> word_prob, scope_name should be clarified inside the func.
            embedding_matrix: embedding_matrix
            vocab_size: size of vocabulary
            cell: rnn cell used in decoder
            ...
        Returns:
            beam_output: generated sequences
            beam_scores: log probability of each sequence
        """

        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[SOS_ID]])
        beam_probs_0 = tf.constant([0.])
    
        cand_seqs_0 = tf.constant([[EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])
        
        """ The beam_step func is written for MultiRNN case, where the states are
        a list of Tensors. Therefore, if the cell is not MultiRNN type, manully 
        convert it into a list.
        """
        if type(cell) is not tf.contrib.rnn.MultiRNNCell:
            states_0 = [init_state]
        else:
            states_0 = init_state
    
        def beam_cond(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            return tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs)
    
        def beam_step(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = tf.nn.embedding_lookup(embedding_matrix, inputs)
            
            """ Specified check if the decoder is Multi-layer RNN. If it is, use the dec_func
            directly. If it is not, remove the list when feeding, put back the [] after processing
            """
            if type(cell) is not tf.contrib.rnn.MultiRNNCell:
                state_output, logprobs2d = dec_step_func(decoder_input, states[0])
                state_output = [state_output]
            else:
                state_output, logprobs2d = dec_step_func(decoder_input, states)
      
            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [batch_size, EOS_ID]),
                tf.tile([[-3e38]], [batch_size, 1]),
                tf.slice(total_probs, [0, EOS_ID + 1], [batch_size, vocab_size - EOS_ID - 1])], 
                axis=1)
      
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size_plh)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)
      
            next_bases = tf.floordiv(top_indices, vocab_size)
            next_mods = tf.mod(top_indices, vocab_size)
      
            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases), 
                tf.reshape(next_mods, [-1, 1])], 1)
      
            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], axis=0)
            EOS_probs = tf.slice(total_probs, [0, EOS_ID], [batch_size, 1])
            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], axis=0)
      
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size_plh)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)
      
            return [time + 1, next_beam_probs, next_beam_seqs, next_cand_probs, next_cand_seqs] + next_states
    
        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None,])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((cand_probs_0, tf.TensorShape([None,])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        #var_shape.extend([(state_0, tf.TensorShape([None, cell.state_size])) for state_0 in states_0])
        """ Specifial check """
        if type(cell) is not tf.contrib.rnn.MultiRNNCell:
            var_shape.append((states_0[0], tf.TensorShape([None, cell.state_size])))
        else:
            for idx, state_0 in enumerate(states_0): 
                var_shape.append((state_0, tf.TensorShape([None, cell.state_size[idx]])))
        loop_vars, loop_var_shapes = zip(* var_shape)
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes, back_prop=False)
    	# time, beam_probs, beam_seqs, cand_probs, cand_seqs, _ = ret_vars
        cand_seqs = ret_vars[4]
        cand_probs = ret_vars[3]
        beam_output = cand_seqs
        beam_scores = cand_probs
        return beam_output, beam_scores
