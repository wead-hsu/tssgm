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

UNK_TOKEN = "$UNK$"
ASP_TOKEN = "$T$"

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
      None
    """
    
    if os.path.exists(os.path.join(data_dir, 'vocab.pkl')):
        logger.info('Processed vocab already exists in {}'.format(data_dir))
        word2idx_sent = pkl.load(open(os.path.join(data_dir, 'vocab.pkl'), 'rb'))
        word2idx_aspect = pkl.load(open(os.path.join(data_dir, 'aspect.pkl'), 'rb'))
    else:
        # keep the same format as in previous work
        words_sent, words_aspect = [], []
        if isinstance(fns, str): fns = [fns]
        for fn in fns:
            data          = pkl.load(open(fn, 'rb'), encoding='latin')
            # MemAbsa
            words_sent   += [w for sample in data for i, w in enumerate(sample['tokens']) if sample['tags'][i] == 'O']
            words_aspect += [' '.join([w for i, w in enumerate(sample['tokens']) if sample['tags'][int(i)] != 'O']) for sample in data]
            # TCLSTM
            #words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])]
        def build_vocab(words, tokens):
            print(len(words))
            words = Counter(words)
            print(len(words))
            word2idx = {token: i for i, token in enumerate(tokens)}
            word2idx.update({w[0]: i+len(tokens) for i, w in enumerate(words.most_common())})
            print(len(word2idx))
            return word2idx
        word2idx_sent = build_vocab(words_sent, [UNK_TOKEN, ASP_TOKEN])
        word2idx_aspect = build_vocab(words_aspect, [])
        with open(os.path.join(data_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2idx_sent, f)
        with open(os.path.join(data_dir, 'aspect.pkl'), 'wb') as f:
            pkl.dump(word2idx_aspect, f)
        logger.info('Vocabuary for input words has been created. shape={}'.format(len(word2idx_sent)))
        logger.info('Vocabuary for aspect words has been created. shape={}'.format(len(word2idx_aspect)))
    
    # create embedding from pretrained vectors
    if os.path.exists(os.path.join(data_dir, 'emb_word.pkl')):
        logger.info('word embedding matrix already exisits in {}'.format(data_dir))
        emb_init_word = pkl.load(open(os.path.join(data_dir, 'emb_word.pkl'), 'rb'))
        emb_init_aspect = pkl.load(open(os.path.join(data_dir, 'emb_aspect.pkl'), 'rb'))
    else:
        if pretrain_fn is None:
            logger.info('Pretrained vector is not given, the embedding matrix is not created')
        else:
            pretrained_vectors = {str(l.split()[0]): [float(n) for n in l.split()[1:]] for l in open(pretrain_fn).readlines()}
            dim_emb = len(pretrained_vectors[list(pretrained_vectors.keys())[0]])
            print(dim_emb)
            def build_emb(pretrained_vectors, word2idx):
                emb_init = np.random.randn(len(word2idx), dim_emb) * 1e-2
                for w in word2idx:
                    if w in pretrained_vectors:
                        emb_init[word2idx[w]] = pretrained_vectors[w]
                return emb_init
            emb_init_word = build_emb(pretrained_vectors, word2idx_sent).astype('float32')
            with open(os.path.join(data_dir, 'emb_word.pkl'), 'wb') as f:
                pkl.dump(emb_init_word, f)
            logger.info('Pretrained vectors has been created from {}'.format(pretrain_fn))
            emb_init_aspect = build_emb(pretrained_vectors, word2idx_aspect).astype('float32')
            with open(os.path.join(data_dir, 'emb_aspect.pkl'), 'wb') as f:
                pkl.dump(emb_init_aspect, f)
            logger.info('Pretrained vectors has been created from {}'.format(pretrain_fn))

    return word2idx_sent, word2idx_aspect, emb_init_word, emb_init_aspect

def load_data(data_dir):
    word2idx = pkl.load(open(os.path.join(data_dir, 'vocab.pkl')))
    embedding = pkl.load(open(os.path.join(data_dir, 'emb.pkl')))
    return word2idx, embedding

class TCLSTM(BaseModel):
    def __init__(self, word2idx, embedding_dim, batch_size, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding):
        super(TCLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        #self.display_step = display_step
        self.word2idx = word2idx
        self.type_ = 'TC'

        if embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(embedding))
        elif isinstance(embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(embedding.shape)))
            self.embedding = tf.constant(embedding, name='embedding')
        elif isinstance(embedding, tf.Variable):
            logger.info('Import tensor as the embedding: '.format(embedding.name))
            self.embedding = embedding

        self.keep_rate_plh = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.y = tf.placeholder(tf.int32, [None, self.n_class])
            self.sen_len = tf.placeholder(tf.int32, None)

            self.x_bw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.y_bw = tf.placeholder(tf.int32, [None, self.n_class])
            self.sen_len_bw = tf.placeholder(tf.int32, [None])

            self.target_words = tf.placeholder(tf.int32, [None, 1])

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

        inputs_fw = tf.nn.embedding_lookup(self.embedding, self.x)
        inputs_bw = tf.nn.embedding_lookup(self.embedding, self.x_bw)
        target = tf.reduce_mean(tf.nn.embedding_lookup(self.embedding, self.target_words), 1, keep_dims=True)
        batch_size = tf.shape(inputs_bw)[0]
        target = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim]) + target
        inputs_fw = tf.concat([inputs_fw, target], 2)
        inputs_bw = tf.concat([inputs_bw, target], 2)
        prob = self.bi_dynamic_lstm(inputs_fw, inputs_bw)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=self.y))
            self.cost = cost

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, axis=1), tf.argmax(self.y, axis=1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            self.accuracy = accuracy
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        summary_loss = tf.summary.scalar('loss', cost)
        summary_acc = tf.summary.scalar('acc', _acc)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])

    def bi_dynamic_lstm(self, inputs_fw, inputs_bw):
        """
        :params: self.x, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=self.keep_rate_plh)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=self.keep_rate_plh)

        with tf.name_scope('forward_lstm'):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=self.sen_len,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.name_scope('backward_lstm'):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=self.sen_len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return predict

    def run(self, sess, train_data, test_data, n_iter, keep_prob):
        with tf.Session() as sess:
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_' +  '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
            train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)
            
            """
            tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word = load_inputs_twitter(
                FLAGS.train_file_path,
                self.word2idx,
                self.max_sentence_len,
                self.type_
            )
            te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word = load_inputs_twitter(
                FLAGS.test_file_path,
                self.word2idx,
                self.max_sentence_len,
                self.type_
            )
            """

            sess.run(tf.global_variables_initializer())

            max_acc = 0.
            for i in range(n_iter):
                for train, _ in self.get_batch_data(train_data, keep_prob):
                    _, step, summary = sess.run([self.optimizer, self.global_step, self.train_summary_op], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                acc, loss, cnt = 0., 0., 0
                for test, num in self.get_batch_data(test_data, 1):
                    _loss, _acc, summary = sess.run([self.cost, self.accuracy, self.test_summary_op], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                print(cnt)
                print(acc)
                test_summary_writer.add_summary(summary, step)
                print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss / cnt, acc / cnt))
                test_summary_writer.add_summary(summary, step)
                if acc / cnt > max_acc:
                    max_acc = acc / cnt
            print('Optimization Finished! Max acc={}'.format(max_acc))

            print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg
            ))

    def prepare_data(self, samples, sentence_len, type_='', encoding='utf8'):
        word_to_id = self.word2idx
    
        x, y, sen_len = [], [], []
        x_r, sen_len_r = [], []
        target_words = []
        y_dict = {'positive': [1,0,0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}
        for sample in samples:
            target_word = [sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O']
            target_word = list(map(lambda w: word_to_id.get(w, 0), target_word))
            target_words.append([target_word[0]]) #?????
            
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

        if type_ == 'TD':
            return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
                   np.asarray(sen_len_r), np.asarray(y)
        elif type_ == 'TC':
            return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
                   np.asarray(sen_len_r), np.asarray(y), np.asarray(target_words)
        else:
            return np.asarray(x), np.asarray(sen_len), np.asarray(y)
    
    def get_batch_data(self, train_data, keep_prob):
        for samples, in train_data:
            x, sen_len, x_bw, sen_len_bw, y, target_words = self.prepare_data(samples, self.max_sentence_len, type_='TC')
            feed_dict = {
                self.x: x,
                self.x_bw: x_bw,
                self.y: y,
                self.sen_len: sen_len,
                self.sen_len_bw: sen_len_bw,
                self.target_words: target_words,
                self.keep_rate_plh: keep_prob,
            }
            yield feed_dict, len(x)

def main(_):

    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open('../../../../data/se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
    test = pkl.load(open('../../../../data/se2014task06/tabsa-rest/test.pkl', 'rb'), encoding='latin')
    
    fns = ['../../../../data/se2014task06/tabsa-rest/train.pkl',
            '../../../../data/se2014task06/tabsa-rest/dev.pkl',
            '../../../../data/se2014task06/tabsa-rest/test.pkl',]
    #data_dir = '.'
    data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, _, embedding, _ = preprocess_data(fns, '/Users/wdxu/data/glove/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()

        model = TCLSTM(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                batch_size=FLAGS.batch_size, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding)

        model.run(sess, train_it, test_it, FLAGS.n_iter, FLAGS.keep_prob)

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
    tf.app.flags.DEFINE_string('keep_prob', 0.5, 'keep rate')

    tf.app.run()
