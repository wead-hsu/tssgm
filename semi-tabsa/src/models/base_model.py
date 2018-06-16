import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import pickle

class BaseModel(object):
    """
    The class is uesd as the base model for other dnn models.
    The model provides interfaces for global_step, train_op, saver, get_feed_dict
    """
    def __init__(self):
        self.init_global_step()
    
    def model_setup(self, args):
        raise NotImplementedError

    def init_global_step(self):
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)
    
    def training_op(self, cost, var_list,
            grad_clip=-1,
            max_norm=-1,
            learning_rate=0.001,
            grads=None):
        
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

    def _get_feed_dict(self, samples):
        raise NotImplementedError

    def _create_saver(self, var_list, max_to_keep=1):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list        = var_list,
                max_to_keep     = max_to_keep,
                write_version   = saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver
