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

class TCDecoder(BaseModel):
    def __init__(self, word2idx, embedding_dim, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, dim_z, decoder_type, grad_clip):
        super(TCDecoder, self).__init__()

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

        if embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(embedding))
        elif isinstance(embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(embedding.shape)))
            self.embedding = tf.constant(embedding, name='embedding')
        elif isinstance(embedding, tf.Tensor):
            logger.info('Import tensor as the embedding: '.format(embedding.name))
            self.embedding = embedding
        else:
            raise Exception('Embedding type {} is not supported'.format(type(embedding)))


    def create_placeholders(self, tag):
        with tf.name_scope('inputs'):
            plhs = dict()
            if tag == 'xa':
                plhs['x_fw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_fw')
                plhs['sen_len_fw']   = tf.placeholder(tf.int32, [None], name='sen_len_fw')
                plhs['x_bw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_bw')
                plhs['sen_len_bw']   = tf.placeholder(tf.int32, [None], name='sen_len_bw')
                plhs['target_words'] = tf.placeholder(tf.int32, [None, 1], name='target_words')
            elif tag == 'y':
                plhs['y']            = tf.placeholder(tf.float32, [None, self.n_class], name='y')
            elif tag == 'hyper':
                plhs['keep_rate']    = tf.placeholder(tf.float32, [], name='keep_rate')
                plhs['is_training']  = tf.placeholder(tf.bool, name='is_training')
            else:
                raise Exception('{} is not supported in create_placeholders'.format(tag))   
        return plhs

    def forward_rnn(self, emb_inp, mask, init_state, y):
        if self.decoder_type.lower() == 'sclstm':
            logger.info('Using ScLSTM as the decoder')
            from src.models.layers.sclstm import ScLSTM
            y_inp = tf.tile(y[:, None, :], [1, tf.shape(emb_inp)[1], 1])
            sclstm_layer = ScLSTM(emb_inp.shape[2], self.n_hidden, self.n_class, cell_clip=self.grad_clip)
            _, dec_outs = sclstm_layer.forward(emb_inp, mask, y_inp, return_final=False, initial_state=(init_state, init_state))
            cell = sclstm_layer._lstm_step
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
        else:
            raise Exception('Decoder type {} is not supported'.format(self.decoder_type))


        w_t = tf.get_variable(
                'proj_w', 
                [self.vocab_size, self.n_hidden])
        b = tf.get_variable(
                "proj_b", 
                [self.vocab_size])
        out_proj = (w_t, b)

        return dec_outs, out_proj, cell

    def create_softmax_layer(self, proj, inp, targets, weights):
        w_t, b = proj
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

    def forward(self, xa_inputs, y_inputs, z, hyper_inputs):
        with tf.name_scope('forward'):
            batch_size = tf.shape(xa_inputs['x_fw'])[0]
       
            z_ext = tf.contrib.layers.fully_connected(tf.reshape(z, [-1, self.dim_z]), self.n_hidden, tf.tanh, scope='extend_z')
            z_ext = tf.nn.dropout(z_ext, hyper_inputs['keep_rate'])
            yz = tf.concat([z_ext, y_inputs['y']], axis=1)
            yz = tf.contrib.layers.fully_connected(yz, self.n_hidden, None, scope='yz_dense')
            yz = tf.contrib.layers.batch_norm(yz, center=True, scale=True, is_training=hyper_inputs['is_training'], scope='yz_bn')
            yz = tf.tanh(yz)
            yz = tf.nn.dropout(yz, hyper_inputs['keep_rate'])

            inputs_fw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_fw'])
            inputs_bw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_bw'])
            inputs_fw = tf.concat([tf.zeros([batch_size, 1, self.embedding_dim]), inputs_fw[:, :-1, :]], axis=1)
            inputs_bw = tf.concat([tf.zeros([batch_size, 1, self.embedding_dim]), inputs_bw[:, :-1:,:]], axis=1)
            target = tf.reduce_mean(tf.nn.embedding_lookup(self.embedding, xa_inputs['target_words']), 1, keep_dims=True)
            target = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim]) + target
            inputs_fw = tf.concat([inputs_fw, target], 2)
            inputs_bw = tf.concat([inputs_bw, target], 2)
            
            with tf.variable_scope('forward_lstm'):
                mask = tf.to_float(tf.sequence_mask(xa_inputs['sen_len_fw'], tf.shape(inputs_fw)[1]))
                outs, proj, cell  = self.forward_rnn(inputs_fw, mask, yz, y_inputs['y'])
                outs = tf.nn.dropout(outs, hyper_inputs['keep_rate'])
                recons_loss_fw = self.create_softmax_layer(proj, outs, xa_inputs['x_fw'], mask) * mask
                recons_loss_fw = tf.reduce_sum(recons_loss_fw, axis=1)

            with tf.variable_scope('backward_lstm'):
                mask = tf.to_float(tf.sequence_mask(xa_inputs['sen_len_bw'], tf.shape(inputs_bw)[1]))
                outs, proj, cell  = self.forward_rnn(inputs_bw, mask, yz, y_inputs['y'])
                outs = tf.nn.dropout(outs, hyper_inputs['keep_rate'])
                recons_loss_bw = self.create_softmax_layer(proj, outs, xa_inputs['x_bw'], mask) * mask
                recons_loss_bw = tf.reduce_sum(recons_loss_bw, axis=1)

            recons_loss = recons_loss_fw + recons_loss_bw
            ppl_fw = tf.exp(tf.reduce_sum(recons_loss_fw) / (tf.to_float(tf.reduce_sum(xa_inputs['sen_len_fw'])) + 1e-3))
            ppl_bw = tf.exp(tf.reduce_sum(recons_loss_bw) / (tf.to_float(tf.reduce_sum(xa_inputs['sen_len_bw'])) + 1e-3))
            ppl = tf.exp(tf.reduce_sum(recons_loss) / (tf.to_float(tf.reduce_sum(xa_inputs['sen_len_fw'] + xa_inputs['sen_len_bw'])) + 1e-3))

            #recons_loss = tf.reduce_mean(recons_loss)

        return recons_loss, ppl_fw, ppl_bw, ppl

    def run(self, sess, train_data, test_data, n_iter, keep_rate, save_dir):
        self.init_global_step()
        xa_inputs = self.create_placeholders('xa')
        y_inputs = self.create_placeholders('y')
        hyper_inputs = self.create_placeholders('hyper')

        z = tf.zeros([tf.shape(xa_inputs['x_fw'])[0], self.dim_z])
        cost, ppl_fw, ppl_bw, ppl = self.forward(xa_inputs, y_inputs, z, hyper_inputs)
        cost = tf.reduce_mean(cost)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)
   
        summary_loss = tf.summary.scalar('loss', cost)
        summary_ppl_fw = tf.summary.scalar('ppl_fw', ppl_fw)
        summary_ppl_bw = tf.summary.scalar('ppl_bw', ppl_bw)
        summary_ppl = tf.summary.scalar('ppl', ppl)

        train_summary_op = tf.summary.merge([summary_loss, summary_ppl_fw, summary_ppl_bw, summary_ppl])
        validate_summary_op = tf.summary.merge([summary_loss, summary_ppl_fw, summary_ppl_bw, summary_ppl])
        test_summary_op = tf.summary.merge([summary_loss, summary_ppl_fw, summary_ppl_bw, summary_ppl])

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

        min_ppl = 1e8
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
                _loss, _ppl, summary = sess.run([cost, ppl, test_summary_op], feed_dict=feed_dict)
                res += _ppl * num
                loss += _loss * num
                cnt += num
            test_summary_writer.add_summary(summary, step)
            print('Iter {}: mini-batch loss={:.6f}, test ppl={:.6f}'.format(step, loss / cnt, res / cnt))
            test_summary_writer.add_summary(summary, step)
            if res / cnt < min_ppl:
                min_ppl = res / cnt
        print('Optimization Finished! Min ppl={}'.format(min_ppl))

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
    
        x, y, sen_len = [], [], []
        x_r, sen_len_r = [], []
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
            words = [sample['tokens'][i] if sample['tags'][i] == 'O' else '$t$' for i in range(len(words))]
            words_l, words_r = [], []
            flag = True
            for word in words:
                if word == '$t$':
                    flag = False
                    continue
                if flag:
                    if word in word_to_id:
                        words_l.append(word_to_id[word])
                else:
                    if word in word_to_id:
                        words_r.append(word_to_id[word])
            type_ = 'TC'
            if type_ == 'TD' or type_ == 'TC':
                words_l.extend(target_word)
                words_l.reverse()
                sen_len.append(len(words_l))
                x.append(words_l + [0] * (sentence_len - len(words_l)))
                words_r = target_word + words_r
                sen_len_r.append(len(words_r))
                x_r.append(words_r + [0] * (sentence_len - len(words_r)))
            else:
                words = words_l + target_word + words_r
                sen_len.append(len(words))
                x.append(words + [0] * (sentence_len - len(words)))

        return {'x_fw': np.asarray(x), 
                'sen_len_fw': np.asarray(sen_len), 
                'x_bw': np.asarray(x_r),
                'sen_len_bw':np.asarray(sen_len_r), 
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

    data_dir = '../classifier/0617'
    #data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, embedding = preprocess_data(fns, '/Users/wdxu/data/glove/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()

        model = TCDecoder(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding,
                dim_z = 3,
                decoder_type=FLAGS.decoder_type,
                grad_clip=FLAGS.grad_clip)

        model.run(sess, train_it, test_it, FLAGS.n_iter, FLAGS.keep_rate, '.')

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
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
