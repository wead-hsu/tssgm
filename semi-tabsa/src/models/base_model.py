import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(object):
    """
    The class is uesd as the base model for other models in the semi-tabsa project
    The model is able to provide a forward function of other components. And it
    can be trained as a single model.
    The model is responsible for:
        1. prepare_data (assume that the format of input raw data is known).
        2. create_placeholders. Since the model is used for a certain task, the model
            should know what kinds of input placeholders should be provided.
        4. forward. Given the inputs, output results for computing the 
            loss or other components.
        5. get_feed_dict. Provided with the raw input data, the model should be
            able to transform the data to the format that can be fed to the input 
            placeholders
        7. saver: the model is aware of the parameters that should be saved.
        8. init_global_step: a unique tf tensor over the entire model when initialized.
        9. training_op: simplified interface for parameter updating
    """
    def __init__(self):
        """
        All hyper-parameters should be provided in the init function.
        """

    def init_global_step(self):
        if not hasattr(self, 'global_step'):
            with tf.device('/cpu:0'):
                self.global_step = tf.get_variable('global_step', [],
                        initializer=tf.constant_initializer(0), trainable=False)
                logger.info('Global step tensor has been created')

    def create_placeholders(self):
        raise NotImplementedError
    
    def forward(self, inputs):
        raise NotImplementedError

    def training_op(self, cost, var_list, grad_clip=-1, max_norm=-1, learning_rate=0.001, grads=None):
        # ------------- calc gradients --------------------------
        if grads is None:
            grads = tf.gradients(cost, var_list)

        for i, g in enumerate(grads):
            if g is None:
                print('WARNING: {} is not in the graph'.format(var_list[i].name))
        grads = [g for g in grads if g is not None]
        if grad_clip > 0: grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in grads]
        if max_norm > 0: grads, _ = tf.clip_by_global_norm(grads, max_norm)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate)

        if not hasattr(self, 'global_step'):
            self.init_global_step()

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def get_feed_dict(self, plhs_dict, data_dict):
        feed_dict = {}
        for plh in plhs_dict:
            if not plh in data_dict:
                raise Exception('{} is not given for the input'.format(plh))
            feed_dict[plhs_dict[plh]] = data_dict[plh]
        return feed_dict

    def _create_saver(self, var_list, max_to_keep=1):
        # -------------- initialization and restore ---------------
        # For distributed version, assign/initialization/saver is not allowed on each GPU
        self.saver = tf.train.Saver(
                var_list        = var_list,
                max_to_keep     = max_to_keep,
                write_version   = saver_pb2.SaverDef.V2)  # save all, including word embeddings
        return self.saver
