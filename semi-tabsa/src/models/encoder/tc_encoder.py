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

class TCEncoder(BaseModel):
    def __init__(self, word2idx, embedding_dim, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, dim_z, grad_clip):
        super(TCEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.word2idx = word2idx
        self.dim_z = dim_z
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


        with tf.name_scope('weights'):
            self.weights = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_w',
                    shape=[2 * self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

    def bi_dynamic_lstm(self, inputs_fw, inputs_bw, sen_len_fw, sen_len_bw, keep_rate):
        """
        :params: self.x_fw, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=keep_rate)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=keep_rate)

        with tf.name_scope('forward_lstm'):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden, cell_clip=self.grad_clip),
                inputs=inputs_fw,
                sequence_length=sen_len_fw,
                dtype=tf.float32,
                scope='LSTM_fw',
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (sen_len_fw - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.name_scope('backward_lstm'):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden, cell_clip=self.grad_clip),
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

    def forward(self, xa_inputs, y_inputs, hyper_inputs):
        with tf.name_scope('forward'):
            inputs_fw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_fw'])
            inputs_bw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_bw'])
            target = tf.reduce_mean(tf.nn.embedding_lookup(self.embedding, xa_inputs['target_words']), 1, keep_dims=True)
            batch_size = tf.shape(inputs_bw)[0]
            target = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim]) + target
            inputs_fw = tf.concat([inputs_fw, target], 2)
            inputs_bw = tf.concat([inputs_bw, target], 2)
            enc_state = self.bi_dynamic_lstm(inputs_fw, inputs_bw, xa_inputs['sen_len_fw'], xa_inputs['sen_len_bw'], hyper_inputs['keep_rate'])

            enc_state = tf.nn.dropout(enc_state, hyper_inputs['keep_rate'])
            enc_state = tf.contrib.layers.fully_connected(enc_state, 
                    num_outputs=self.n_hidden,
                    activation_fn=None,
                    scope='x_to_a') # change to layers.dense?
            enc_state = tf.contrib.layers.batch_norm(enc_state,
                    center=True,
                    scale=True,
                    is_training=hyper_inputs['is_training'],
                    scope='bn_a')
            enc_state = tf.tanh(enc_state)
            enc_state = tf.nn.dropout(enc_state, hyper_inputs['keep_rate'])
    
            with tf.variable_scope('latent'):
                y_enc_in = tf.contrib.layers.fully_connected(y_inputs['y'], self.n_hidden, tf.tanh, scope='y_enc_in')
                y_enc_in = tf.nn.dropout(y_enc_in, hyper_inputs['keep_rate'])
                pst_in = tf.concat([y_enc_in, enc_state], axis=1)
                pst_in = tf.contrib.layers.fully_connected(pst_in, self.n_hidden, None, scope='pst_in_dense')
                pst_in = tf.contrib.layers.batch_norm(pst_in, center=True, scale=True, 
                        is_training=hyper_inputs['is_training'], scope='pst_in_bn')
                pst_in = tf.tanh(pst_in)
                pst_in = tf.nn.dropout(pst_in, hyper_inputs['keep_rate'])
                mu_pst = tf.contrib.layers.fully_connected(pst_in, self.dim_z, None, 
                        scope='mu_posterior')
                logvar_pst = tf.contrib.layers.fully_connected(pst_in, self.dim_z, None,
                        scope='logvar_posterior')
                mu_pri = tf.zeros_like(mu_pst)
                logvar_pri = tf.ones_like(logvar_pst)

                dist_pri = tf.contrib.distributions.Normal(loc=mu_pri, scale=tf.exp(logvar_pri))
                dist_pst = tf.contrib.distributions.Normal(loc=mu_pst, scale=tf.exp(logvar_pst))
                kl_loss = tf.contrib.distributions.kl_divergence(dist_pst, dist_pri)
                kl_loss = tf.reduce_sum(kl_loss, axis=1)
                
                z_st_pri = dist_pri.sample()
                z_st_pst = dist_pst.sample()
        return z_st_pst, z_st_pri, kl_loss

    def run(self, sess, train_data, test_data, n_iter, keep_rate, save_dir):
        self.init_global_step()
        xa_inputs = self.create_placeholders('xa')
        y_inputs = self.create_placeholders('y')
        hyper_inputs = self.create_placeholders('hyper')

        logits, _, _ = self.forward(xa_inputs, y_inputs, hyper_inputs)
        y = tf.placeholder(tf.float32, [None, self.n_class], 'y')
        print(logits, y)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        summary_loss = tf.summary.scalar('loss', cost)
        summary_acc = tf.summary.scalar('acc', _acc)
        train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
        test_summary_op = tf.summary.merge([summary_loss, summary_acc])

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

        max_acc = 0.
        for i in range(n_iter):
            #for train, _ in self.get_batch_data(train_data, keep_rate):
            for samples, in train_data:
                feed_data = self.prepare_data(samples)
                feed_data.update({'keep_rate': keep_rate, 'is_training': True})
                xa_inputs.update(hyper_inputs)
                xa_inputs.update(y_inputs)
                feed_dict = self.get_feed_dict(xa_inputs, feed_data)
                feed_dict.update({y: get_y(samples)})
                
                _, step, summary = sess.run([optimizer, self.global_step, train_summary_op], feed_dict=feed_dict)
                train_summary_writer.add_summary(summary, step)
            acc, loss, cnt = 0., 0., 0
            for samples, in test_data:
                feed_data = self.prepare_data(samples)
                feed_data.update({'keep_rate': 1.0, 'is_training': False})
                xa_inputs.update(hyper_inputs)
                xa_inputs.update(y_inputs)
                feed_dict = self.get_feed_dict(xa_inputs, feed_data)
                feed_dict.update({y: get_y(samples)})

                num = len(samples)
                _loss, _acc, summary = sess.run([cost, accuracy, test_summary_op], feed_dict=feed_dict)
                acc += _acc
                loss += _loss * num
                cnt += num
            #print(cnt)
            #print(acc)
            test_summary_writer.add_summary(summary, step)
            print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss / cnt, acc / cnt))
            test_summary_writer.add_summary(summary, step)
            if acc / cnt > max_acc:
                max_acc = acc / cnt
        print('Optimization Finished! Max acc={}'.format(max_acc))
    
        print('Learning_rate={}, iter_num={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                n_iter,
                self.n_hidden,
                self.l2_reg
            ))

    def prepare_data(self, samples): #, sentence_len, type_='', encoding='utf8'):
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
            if type_ == 'TD' or type_ == 'TC':
                words_l.extend(target_word)
                sen_len.append(len(words_l))
                x.append(words_l + [0] * (sentence_len - len(words_l)))
                tmp = target_word + words_r
                tmp.reverse()
                sen_len_r.append(len(tmp))
                x_r.append(tmp + [0] * (sentence_len - len(tmp)))
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

        model = TCEncoder(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding,
                dim_z = 3,
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
    tf.app.flags.DEFINE_float('grad_clip', 5.0, 'gradient_clip')

    tf.app.run()
