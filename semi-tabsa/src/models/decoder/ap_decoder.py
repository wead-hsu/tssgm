import sys
sys.path.append('../../../')
from src.models.base_model import BaseModel
from collections import Counter
import tensorflow as tf
import logging
import os
import pickle as pkl
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNK_TOKEN = "$unk$"
ASP_TOKEN = "$t$"

def preprocess_data(fns, pretrain_fn, data_dir):
    """
    preprocess the data for tclstm
    if the data has been created, do nothing
    else create vocab and emb
    Args:
        fns: input files
        pretrain_fn: filename for pretrained word vectors
        data_dir: dirname for processed data
    Return:
        word2idx_sent: dict, 
        emb_init_sent: numpy matrix 
    """
    
    if os.path.exists(os.path.join(data_dir, 'vocab_sent.pkl')):
        logger.info('Processed vocab already exists in {}'.format(data_dir))
        word2idx_sent = pkl.load(open(os.path.join(data_dir, 'vocab_sent.pkl'), 'rb'))
    else:
        # keep the same format as in previous work
        words_sent = []
        if isinstance(fns, str): fns = [fns]
        for fn in fns:
            data          = pkl.load(open(fn, 'rb'), encoding='latin')
            words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])]

        def build_vocab(words, tokens):
            words = Counter(words)
            word2idx = {token: i for i, token in enumerate(tokens)}
            word2idx.update({w[0]: i+len(tokens) for i, w in enumerate(words.most_common())})
            return word2idx
        word2idx_sent = build_vocab(words_sent, [UNK_TOKEN, ASP_TOKEN])
        with open(os.path.join(data_dir, 'vocab_sent.pkl'), 'wb') as f:
            pkl.dump(word2idx_sent, f)
        logger.info('Vocabuary for input words has been created. shape={}'.format(len(word2idx_sent)))
    
    # create embedding from pretrained vectors
    if os.path.exists(os.path.join(data_dir, 'emb_sent.pkl')):
        logger.info('word embedding matrix already exisits in {}'.format(data_dir))
        emb_init_sent = pkl.load(open(os.path.join(data_dir, 'emb_sent.pkl'), 'rb'))
    else:
        if pretrain_fn is None:
            logger.info('Pretrained vector is not given, the embedding matrix is not created')
        else:
            pretrained_vectors = {str(l.split()[0]): [float(n) for n in l.split()[1:]] for l in open(pretrain_fn).readlines()}
            dim_emb = len(pretrained_vectors[list(pretrained_vectors.keys())[0]])
            def build_emb(pretrained_vectors, word2idx):
                emb_init = np.random.randn(len(word2idx), dim_emb) * 1e-2
                for w in word2idx:
                    if w in pretrained_vectors:
                        emb_init[word2idx[w]] = pretrained_vectors[w]
                return emb_init
            emb_init_sent = build_emb(pretrained_vectors, word2idx_sent).astype('float32')
            with open(os.path.join(data_dir, 'emb_sent.pkl'), 'wb') as f:
                pkl.dump(emb_init_sent, f)
            logger.info('Pretrained vectors has been created from {}'.format(pretrain_fn))
    
    return word2idx_sent, emb_init_sent

def load_data(data_dir):
    word2idx = pkl.load(open(os.path.join(data_dir, 'vocab.pkl')))
    embedding = pkl.load(open(os.path.join(data_dir, 'emb.pkl')))
    return word2idx, embedding

class APDecoder(BaseModel):
    def __init__(self, word2idx, embedding_dim, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, dim_z, decoder_type, grad_clip, position, bidirection, sharefc=True):
        super(APDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.word2idx = word2idx
        self.vocab_size = max(word2idx.values()) + 1
        self.dim_z = dim_z
        self.decoder_type = decoder_type
        self.grad_clip = grad_clip
        self.position = position
        self.bidirection = bidirection
        self.sharefc = sharefc
        self.use_crf = True

        if embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(wemb_init))
        elif isinstance(embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(embedding.shape)))
            self.embedding = tf.constant(embedding, name='embedding')
        elif isinstance(embedding, tf.Tensor) or isinstance(embedding, tf.Variable):
            logger.info('Import tensor as the embedding: '.format(embedding.name))
            self.embedding = embedding
        else:
            raise Exception('Embedding type {} is not supported'.format(type(embedding)))

        if self.position == 'binary':
            logger.info('Binary embedding is initialized.')
            wemb_init = np.random.randn(2, self.embedding_dim//10) * 1e-2
            self.pos_embedding = tf.get_variable('pos_embedding', [2, embedding_dim/10], initializer=tf.constant_initializer(wemb_init))
        elif self.position == 'distance':
            logger.info('Distance embedding is initialized and trainable')
            num_units = self.embedding_dim // 10
            position_enc = np.zeros([2*self.max_sentence_len, num_units])
            for i in range(-self.max_sentence_len, self.max_sentence_len):
                for j in range(num_units):
                    if j % 2 == 0:
                        position_enc[i+self.max_sentence_len, j] = np.sin(i/np.power(10000, j/num_units))
                    else:
                        position_enc[i+self.max_sentence_len, j] = np.cos(i/np.power(10000, (j//2)*2/num_units))
            self.pos_embedding = tf.get_variable('pos_embedding', [2*self.max_sentence_len, num_units], initializer=tf.constant_initializer(position_enc), trainable=False)
        elif self.position == 'distance-add':
            logger.info('Distance embedding is initialized and trainable')
            num_units = self.embedding_dim
            position_enc = np.zeros([2*self.max_sentence_len, num_units])
            for i in range(-self.max_sentence_len, self.max_sentence_len):
                for j in range(num_units):
                    if j % 2 == 0:
                        position_enc[i+self.max_sentence_len, j] = np.sin(i/np.power(10000, j/num_units))
                    else:
                        position_enc[i+self.max_sentence_len, j] = np.cos(i/np.power(10000, (j//2)*2/num_units))
            self.pos_embedding = tf.get_variable('pos_embedding', [2*self.max_sentence_len, num_units], initializer=tf.constant_initializer(position_enc), trainable=False)

    def create_placeholders(self, tag):
        with tf.name_scope('inputs'):
            plhs = dict()
            if tag == 'xa':
                if self.bidirection:
                    plhs['x_fw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_fw')
                    plhs['x_bw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_bw')
                    plhs['m_fw']         = tf.placeholder(tf.float32, [None, self.max_sentence_len], name='m_fw')
                    plhs['m_bw']         = tf.placeholder(tf.float32, [None, self.max_sentence_len], name='m_bw')
                    plhs['l_fw']         = tf.placeholder(tf.int32, [None], name='l_fw')
                    plhs['l_bw']         = tf.placeholder(tf.int32, [None], name='l_bw')
                    plhs['target_words'] = tf.placeholder(tf.int32, [None, 1], name='target_words')
                    if self.position:
                        plhs['p_fw']     = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='p_fw')
                        plhs['p_bw']     = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='p_bw')
                else:
                    plhs['x']            = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
                    plhs['m']            = tf.placeholder(tf.float32, [None, self.max_sentence_len], name='m')
                    plhs['l']            = tf.placeholder(tf.int32, [None], name='l')
                    plhs['target_words'] = tf.placeholder(tf.int32, [None, 1], name='target_words')
                    if self.position:
                        plhs['p']        = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='p')

            elif tag == 'y':
                plhs['y']            = tf.placeholder(tf.float32, [None, self.n_class], name='y')
            elif tag == 'hyper':
                plhs['keep_rate']    = tf.placeholder(tf.float32, [], name='keep_rate')
                plhs['is_training']  = tf.placeholder(tf.bool, name='is_training')
            else:
                raise Exception('{} is not supported in create_placeholders'.format(tag))   
        return plhs

    def fill_blank(self, mask):
        idx = tf.argmax(mask, axis=1)
        fill_mat = tf.convert_to_tensor(np.tri(self.max_sentence_len, k=-1).astype('float32'))
        fill = tf.gather(fill_mat, idx)
        return mask + fill

    def concat_pos_embedding(self, inputs, p):
        if self.position == 'binary':
            inputs_pos = tf.nn.embedding_lookup(self.pos_embedding, p)
            inputs = tf.concat([inputs, inputs_pos], axis=2)
        elif self.position == 'distance':
            inputs_pos = tf.nn.embedding_lookup(self.pos_embedding, p + self.max_sentence_len)
            inputs = tf.concat([inputs, inputs_pos], axis=2)
        elif self.position == 'distance-add':
            inputs_pos = tf.nn.embedding_lookup(self.pos_embedding, p + self.max_sentence_len)
            inputs = inputs + inputs_pos
        return inputs

    def forward_rnn(self, emb_inp, mask, init_state, y):
        if self.decoder_type.lower() == 'sclstm':
            logger.info('Using ScLSTM as the decoder')
            from src.models.layers.sclstm import ScLSTM
            y_inp = tf.tile(y[:, None, :], [1, tf.shape(emb_inp)[1], 1])
            sclstm_layer = ScLSTM(emb_inp.shape[2], self.n_hidden, self.n_class, cell_clip=self.grad_clip)
            _, dec_outs = sclstm_layer.forward(emb_inp, mask, y_inp, return_final=False, initial_state=(init_state, init_state))
            cell = sclstm_layer._lstm_step
        elif self.decoder_type.lower() == 'fclstm':
            logger.info('Using FcLSTM as the decoder')
            from src.models.layers.fclstm import FcLSTM
            y_inp = tf.tile(y[:, None, :], [1, tf.shape(emb_inp)[1], 1])
            fclstm_layer = FcLSTM(emb_inp.shape[2], self.n_hidden, self.n_class, cell_clip=self.grad_clip)
            _, dec_outs = fclstm_layer.forward(emb_inp, mask, y_inp, return_final=False, initial_state=(init_state, init_state))
            cell = fclstm_layer._lstm_step
        elif self.decoder_type.lower() == 'lstm':
            logger.info('Using LSTM as the decoder')
            emb_inp = tf.concat([emb_inp, tf.tile(y[:, None, :], [1, tf.shape(emb_inp)[1], 1])], axis=2)
            cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True, cell_clip=self.grad_clip)

            dec_outs, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=emb_inp,
                    sequence_length=tf.to_int32(tf.reduce_sum(mask, axis=1)),
                    initial_state=tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
                    )

        elif self.decoder_type.lower() == 'gelstm':
            logger.info('Using GeLSTM as the decoder')
            cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True, cell_clip=self.grad_clip)

            dec_outs, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=emb_inp,
                    sequence_length=tf.to_int32(tf.reduce_sum(mask, axis=1)),
                    initial_state=tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
                    )

            y_inp = tf.tile(y[:, None, :], [1, tf.shape(emb_inp)[1], 1])
            dec_outs = tf.concat([dec_outs, y_inp], axis=2)
        else:
            raise Exception('Decoder type {} is not supported'.format(self.decoder_type))
        
        """
        dec_outs_flt = tf.reshape(dec_outs, [-1, int(dec_outs.shape[-1])])
        logits_flt = tf.contrib.layers.fully_connected(dec_outs_flt, self.vocab_size, None, weights_initializer=tf.glorot_uniform_initializer(), biases_initializer=tf.glorot_uniform_initializer(), scope='dense')
        logits = tf.reshape(logits_flt, [tf.shape(dec_outs)[0], -1, self.vocab_size])

        return logits
        """

        return dec_outs

    def create_softmax_layer(self, logits, targets, weights, y):
        inp = logits
        w_t = tf.get_variable(
                'proj_w', 
                [self.vocab_size, int(logits.shape[-1])],
                )

        use_depb = False
        if use_depb:
            b = tf.get_variable(
                    "proj_b", 
                    [self.n_class, self.vocab_size],
                    )
            inp_flt = tf.reshape(inp, [-1, tf.shape(inp)[-1]])
            b = tf.nn.embedding_lookup(b, tf.argmax(y, axis=1))
            b = tf.tile(b[:, None, :], [1, tf.shape(inp)[1], 1])
            b = tf.reshape(b, [-1, self.vocab_size])
            logits_flt = tf.add(tf.matmul(inp_flt, w_t, transpose_b=True), b)
        else:
            b = tf.get_variable(
                    "prob_b",
                    [self.vocab_size],
                    )
            inp_flt = tf.reshape(inp, [-1, tf.shape(inp)[-1]])
            logits_flt = tf.add(tf.matmul(inp_flt, w_t, transpose_b=True), b[None, :])

        logits = tf.reshape(logits_flt, [tf.shape(inp_flt)[0], -1, self.vocab_size])

        llh_precise = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=targets,
                weights=weights,
                average_across_timesteps=False,
                average_across_batch=False,
                softmax_loss_function=None)

        llh_precise = tf.reshape(llh_precise, tf.shape(targets))
        return llh_precise

    def create_seqtag_layer(self, logits, targets, sequence_lengths, y):
        """Defines the loss"""
        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logits, tf.to_int32(targets), sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            #loss = tf.reduce_mean(-log_likelihood)
            loss = -log_likelihood
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=targets)
            mask = tf.sequence_mask(sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            loss = tf.reduce_sum(losses, axis=1)

        return loss

    def forward(self, xa_inputs, y_inputs, z, hyper_inputs):
        if self.bidirection:
            return self.forward_bidirection(xa_inputs, y_inputs, z, hyper_inputs)
        else:
            return self.forward_unidirection(xa_inputs, y_inputs, z, hyper_inputs)

    def forward_unidirection(self, xa_inputs, y_inputs, z, hyper_inputs):
        with tf.name_scope('forward'):
            batch_size = tf.shape(xa_inputs['x'])[0]
       
            y = tf.contrib.layers.fully_connected(y_inputs['y'], self.n_hidden, tf.sigmoid, scope='yz_dense')

            inputs = tf.nn.embedding_lookup(self.embedding, xa_inputs['x'])
            target = tf.reduce_mean(tf.nn.embedding_lookup(self.embedding, xa_inputs['target_words']), 1, keep_dims=True)
            inputs = tf.concat([tf.zeros([batch_size, 1, self.embedding_dim]), inputs[:, :-1, :]], axis=1)
            #inputs = tf.concat([target, inputs[:, :-1, :]], axis=1)
           
            if self.position: inputs = self.concat_pos_embedding(inputs, xa_inputs['p'])
            with tf.variable_scope('lstm'):
                outs  = self.forward_rnn(inputs, tf.sequence_mask(xa_inputs['l']), y, y_inputs['y'])
                outs = tf.nn.dropout(outs, hyper_inputs['keep_rate'])
                recons_loss = self.create_seqtag_layer(outs, 1 - tf.to_int64(xa_inputs['m']), xa_inputs['l'], y_inputs['y'])
                #recons_loss = tf.reduce_sum(recons_loss, axis=1)

        return recons_loss

    def forward_bidirection(self, xa_inputs, y_inputs, z, hyper_inputs):
        with tf.name_scope('forward'):
            batch_size = tf.shape(xa_inputs['x_fw'])[0]
       
            y = tf.contrib.layers.fully_connected(y_inputs['y'], self.n_hidden, tf.sigmoid, scope='yz_dense')

            inputs_fw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_fw'])
            inputs_bw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_bw'])
            inputs_fw = tf.concat([tf.zeros([batch_size, 1, self.embedding_dim]), inputs_fw[:, :-1, :]], axis=1)
            inputs_bw = tf.concat([tf.zeros([batch_size, 1, self.embedding_dim]), inputs_bw[:, :-1:,:]], axis=1)
            #target = tf.reduce_mean(tf.nn.embedding_lookup(self.embedding, xa_inputs['target_words']), 1, keep_dims=True)
            #inputs_fw = tf.concat([target, inputs_fw[:, :-1, :]], axis=1)
            #inputs_bw = tf.concat([target, inputs_bw[:, :-1:,:]], axis=1)
            #target = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim]) + target
            #inputs_fw = tf.concat([inputs_fw, target], 2)
            #inputs_bw = tf.concat([inputs_bw, target], 2)

            if self.position: inputs_fw = self.concat_pos_embedding(inputs_fw, xa_inputs['p_fw'])
            if self.position: inputs_bw = self.concat_pos_embedding(inputs_bw, xa_inputs['p_bw'])
            
            with tf.variable_scope('forward_lstm'):
                #mask = tf.to_float(tf.sequence_mask(xa_inputs['sen_len_fw'], tf.shape(inputs_fw)[1]))
                outs_fw = self.forward_rnn(inputs_fw, tf.sequence_mask(xa_inputs['l_fw']), y, y_inputs['y'])
                outs_fw = tf.nn.dropout(outs_fw, hyper_inputs['keep_rate'])
            with tf.variable_scope('backward_lstm'):
                #mask = tf.to_float(tf.sequence_mask(xa_inputs['sen_len_bw'], tf.shape(inputs_bw)[1]))
                outs_bw = self.forward_rnn(inputs_bw, tf.sequence_mask(xa_inputs['l_bw']), y, y_inputs['y'])
                outs_bw = tf.nn.dropout(outs_bw, hyper_inputs['keep_rate'])
            
            if self.sharefc:
                with tf.variable_scope('softmax'):
                    recons_loss_fw = self.create_seqtag_layer(outs_fw, 1 - tf.to_int64(xa_inputs['m_fw']), xa_inputs['l_fw'], y_inputs['y'])
                    #recons_loss_fw = tf.reduce_sum(recons_loss_fw, axis=1)
                with tf.variable_scope('softmax', reuse=True):
                    recons_loss_bw = self.create_seqtag_layer(outs_bw, 1 - tf.to_int64(xa_inputs['m_bw']), xa_inputs['l_bw'], y_inputs['y'])
                    #recons_loss_bw = tf.reduce_sum(recons_loss_bw, axis=1)
            else:
                with tf.variable_scope('softmax_fw'):
                    recons_loss_fw = self.create_seqtag_layer(outs_fw, 1 - tf.to_int64(xa_inputs['m_fw']), xa_inputs['l_fw'], y_inputs['y'])
                    #recons_loss_fw = tf.reduce_sum(recons_loss_fw, axis=1)
                with tf.variable_scope('softmax_bw'):
                    recons_loss_bw = self.create_seqtag_layer(outs_bw, 1 - tf.to_int64(xa_inputs['m_bw']), xa_inputs['l_bw'], y_inputs['y'])
                    #recons_loss_bw = tf.reduce_sum(recons_loss_bw, axis=1)

            recons_loss = recons_loss_fw + recons_loss_bw

        return recons_loss

    def run(self, sess, train_data, test_data, batch_size, n_iter, keep_rate, save_dir):
        self.batch_size = batch_size
        self.init_global_step()
        xa_inputs = self.create_placeholders('xa')
        y_inputs = self.create_placeholders('y')
        hyper_inputs = self.create_placeholders('hyper')
        
        _var = list(xa_inputs.values())[0]
        z = tf.zeros([tf.shape(_var)[0], self.dim_z])
        cost = self.forward(xa_inputs, y_inputs, z, hyper_inputs)
        cost = tf.reduce_mean(cost)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)
   
        summary_loss = tf.summary.scalar('loss', cost)

        train_summary_op = tf.summary.merge([summary_loss])
        validate_summary_op = tf.summary.merge([summary_loss])
        test_summary_op = tf.summary.merge([summary_loss])

        import time
        timestamp = str(int(time.time()))
        _dir = save_dir + '/logs/' + str(timestamp) + '_' +  '_r' + str(self.learning_rate) + '_l' + str(self.l2_reg)
        train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
        test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
        validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        def get_y(samples):
            y_dict = {'positive': [1,0,0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}
            ys = [y_dict[sample['polarity']] for sample in samples]
            return ys

        for i in range(n_iter):
            #for train, _ in self.get_batch_data(train_data, keep_rate):
            for samples, in train_data:
                feed_data = self.prepare_data(samples)
                feed_data.update({'keep_rate': keep_rate, 'is_training': True})
                xa_inputs.update(hyper_inputs)
                xa_inputs.update(y_inputs)
                feed_dict = self.get_feed_dict(xa_inputs, feed_data)

                _, step, summary = sess.run([optimizer, self.global_step, train_summary_op], feed_dict=feed_dict)
                train_summary_writer.add_summary(summary, step)
            res, loss, cnt = 0., 0., 0
            for samples, in test_data:
                feed_data = self.prepare_data(samples)
                feed_data.update({'keep_rate': 1.0, 'is_training': False})
                xa_inputs.update(hyper_inputs)
                xa_inputs.update(y_inputs)
                feed_dict = self.get_feed_dict(xa_inputs, feed_data)
                
                num = len(samples)
                _loss, summary = sess.run([cost, test_summary_op], feed_dict=feed_dict)
                loss += _loss * num
                cnt += num
            test_summary_writer.add_summary(summary, step)
            print('Iter {}: mini-batch loss={:.6f}'.format(step, loss / cnt))
            test_summary_writer.add_summary(summary, step)

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            self.learning_rate,
            n_iter,
            self.batch_size,
            self.n_hidden,
            self.l2_reg
        ))

    def prepare_data(self, samples):
        sentence_len = self.max_sentence_len
        type_ = 'TC'
        encoding = 'utf8'
        word_to_id = self.word2idx
    
        x, y, m, p, l = [], [], [], [], []
        x_r, m_r, p_r, l_r= [], [], [], []
        target_words = []
        y_dict = {'positive': [1,0,0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}
        for sample in samples:
            target_word = [sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O']
            target_word = list(map(lambda w: word_to_id.get(w, 0), target_word))
            target_words.append([target_word[0]]) #?????
            
            if 'polarity' in sample:
                polarity = sample['polarity']
                y.append(y_dict[polarity])
    
            words = sample['tokens']
            words_l, words_r = [], []
            flag = True
            for idx in range(len(words)):
                word = words[idx]
                if sample['tags'][idx] != 'O':
                    flag = False
                    continue
                if flag:
                    words_l.append(word_to_id.get(word, word_to_id[UNK_TOKEN]))
                else:
                    words_r.append(word_to_id.get(word, word_to_id[UNK_TOKEN]))

            if self.bidirection:
                if self.position == 'binary':
                    _p = [1] * len(target_word) + [0] * (self.max_sentence_len - len(target_word))
                    p.append(_p)
                    p_r.append(_p)
                elif self.position == 'distance' or self.position == 'distance-add':
                    _p = [0] * len(target_word) + list(range(1, 1+len(words_l)))
                    _p = _p + [0] * (self.max_sentence_len - len(_p))
                    p.append(_p)
                    _p = [0] * len(target_word) + list(range(1, 1+len(words_r)))
                    _p = _p + [0] * (self.max_sentence_len - len(_p))
                    p_r.append(_p)

                _m = [0.] * len(target_word) + [1.] * len(words_l)
                _m = _m + [0.] * (self.max_sentence_len - len(_m))
                m.append(_m)

                words_l.extend(target_word)
                words_l.reverse()
                x.append(words_l + [0] * (sentence_len - len(words_l)))
                l.append(len(words_l))

                _m_r = [0] * len(target_word) + [1] * len(words_r)
                _m_r = _m_r + [0] * (self.max_sentence_len - len(_m_r))
                m_r.append(_m_r)

                words_r = target_word + words_r
                x_r.append(words_r + [0] * (sentence_len - len(words_r)))
                l_r.append(len(words_r))
            else:
                ml = 0
                words = target_word[:ml] + [1] + words_l + target_word + words_r
                x.append(words + [0] * (sentence_len - len(words)))
                m.append([0.] * len(target_word[:ml]) + [0.] + [1.0 if tag == 'O' else 0.0 for tag in sample['tags']] + [0.] * (sentence_len - len(words)))
                l.append(len(words))

                if self.position == 'binary':
                    p.append([0]*len(target_word[:ml]) + [0] + [1 if tag != 'O' else 0 for tag in sample['tags']] + [0] * (sentence_len - len(words)))
                elif self.position == 'distance' or self.position == 'distance-add':
                    p.append([0]*len(target_word[:ml]) + [0] + list(range(-len(words_l), 0)) + [0]*len(target_word) + list(range(1, 1+len(words_r))) + [0] * (sentence_len - len(words)))
        
        if self.bidirection:
            return {'x_fw': np.asarray(x), 
                    'm_fw': np.asarray(m), 
                    'l_fw': np.asarray(l),
                    'p_fw': np.clip(np.asarray(p), -10, 10),
                    'x_bw': np.asarray(x_r),
                    'm_bw': np.asarray(m_r), 
                    'l_bw': np.asarray(l_r),
                    'p_bw': np.clip(np.asarray(p_r), -10, 10),
                    'target_words': np.asarray(target_words),
                    'y': np.asarray(y),
                    }
        else:
            return {'x': np.asarray(x),
                    'm': np.asarray(m),
                    'l': np.asarray(l),
                    'p': np.clip(np.asarray(p), -10, 10),
                    'target_words': np.asarray(target_words),
                    'y': np.asarray(y),
                    }

def main(_):
    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open('../../../../data/se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
    test = pkl.load(open('../../../../data/se2014task06/tabsa-rest/test.pkl', 'rb'), encoding='latin')
    
    fns = ['../../../../data/se2014task06/tabsa-rest/train.pkl',
            '../../../../data/se2014task06/tabsa-rest/dev.pkl',
            '../../../../data/se2014task06/tabsa-rest/test.pkl',]

    data_dir = '../classifier/data//0617'
    #data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, embedding = preprocess_data(fns, '/Users/wdxu/data/glove/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()

        model = APDecoder(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len +4, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding,
                dim_z = 3,
                #decoder_type=FLAGS.decoder_type,
                decoder_type='sclstm',
                grad_clip=FLAGS.grad_clip,
                position='distance-add',
                bidirection=True)

        model.run(sess, train_it, test_it, FLAGS.batch_size, FLAGS.n_iter, FLAGS.keep_rate, '.')

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 20, 'number of train iter')
    
    tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/train.raw', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/validate.raw', 'validating file')
    tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/test.raw', 'testing file')
    #tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_word_embedding_partial_100.txt', 'embedding file')
    #tf.app.flags.DEFINE_string('word_id_file_path', 'data/twitter/word_id.txt', 'word-id mapping file')
    tf.app.flags.DEFINE_string('type', 'TC', 'model type: ''(default), TD or TC')
    tf.app.flags.DEFINE_float('keep_rate', 0.5, 'keep rate')
    tf.app.flags.DEFINE_string('decoder_type', 'sclstm', '[sclstm, lstm]')
    tf.app.flags.DEFINE_float('grad_clip', 5, 'gradient_clip, <0 == None')

    tf.app.run()



    """
    def bi_dynamic_lstm(self, inputs_fw, inputs_bw, sen_len_fw, sen_len_bw, keep_rate):
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=keep_rate)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=keep_rate)
        with tf.name_scope('forward_lstm'):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=sen_len_fw,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (sen_len_fw - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden


        out`

        with tf.name_scope('backward_lstm'):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=sen_len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return predict
    """
