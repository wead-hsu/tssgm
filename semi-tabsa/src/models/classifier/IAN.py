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

class IANClassifier(BaseModel):
    def __init__(self, word2idx, embedding_dim, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, grad_clip):
        super(IANClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.word2idx = word2idx
        self.grad_clip = grad_clip

        if embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(wemb_init))
        elif isinstance(embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(embedding.shape)))
            self.embedding = tf.constant(embedding, name='embedding')
            #self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(embedding))
        elif isinstance(embedding, tf.Tensor) or isinstance(embedding, tf.Variable):
            logger.info('Import tensor as the embedding: '.format(embedding.name))
            self.embedding = embedding
        else:
            raise Exception('Embedding type is unknow'.format(type(embedding)))

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

        self.W_att = tf.get_variable(
            name='W_att',
            shape=[self.n_hidden + self.embedding_dim, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.b_att = tf.get_variable(
            name='b_att',
            shape=[self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.U = tf.get_variable(
            name='U',
            shape=[self.n_hidden, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.W_p = tf.get_variable(
            name='W_p',
            shape=[ self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.b_p = tf.get_variable(
            name='b_p',
            shape=[self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )


    def create_placeholders(self, tag):
        with tf.name_scope('inputs'):
            plhs = dict()
            if tag == 'xa':
                plhs['x_fw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_fw')
                plhs['x']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
                plhs['x_bw']         = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x_bw')
                plhs['target_words'] = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='target_words')
                plhs['sen_len_fw']   = tf.placeholder(tf.int32, [None], name='sen_len_fw')
                plhs['sen_len_bw']   = tf.placeholder(tf.int32, [None], name='sen_len_bw')
                plhs['sen_len']   = tf.placeholder(tf.int32, [None], name='sen_len')
                plhs['target_len']   = tf.placeholder(tf.int32, [None], name='target_len')
            elif tag == 'y':
                plhs['y']            = tf.placeholder(tf.float32, [None, self.n_class], name='y')
            elif tag == 'hyper':
                plhs['keep_rate']    = tf.placeholder(tf.float32, [], name='keep_rate')
            else:
                raise Exception('{} is not supported in create_placeholders'.format(tag))   
        return plhs
  
    def dynamic_rnn(self, inputs, length, max_len, scope_name, is_reuse=False, out_type='all'):
        with tf.variable_scope("direction_lstm", reuse=is_reuse) as IAN_scope:
            cell = tf.nn.rnn_cell.LSTMCell
            outputs, state = tf.nn.dynamic_rnn(
                 cell(self.n_hidden),
                 inputs=inputs,
                 sequence_length=length,
                 dtype=tf.float32,
                 scope=scope_name
                 )  # outputs -> batch_size * max_len * n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        elif out_type == 'all_avg':
            outputs = tf.reduce_mean(outputs, length)
        return outputs

    def bi_dynamic_rnn(self, inputs, length, max_len, scope_name, is_reuse=False, out_type='all'):
        with tf.variable_scope("bidirection_lstm", reuse=is_reuse) as IAN_scope:
            cell = tf.nn.rnn_cell.LSTMCell
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell(self.n_hidden),
                cell_bw=cell(self.n_hidden),
                inputs=inputs,
                sequence_length=length,
                dtype=tf.float32,
                scope=scope_name
            )
        if out_type == 'last':
            outputs_fw, outputs_bw = outputs
            outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
            outputs = tf.concat([outputs_fw, outputs_bw], 2)
        else:
            outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, 2 * self.n_hidden]), index)  # batch_size * 2n_hidden
        elif out_type == 'all_avg':
            outputs = self.reduce_mean(outputs, length)  # batch_size * 2n_hidden
        return outputs

    def softmax(self, inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def AT(self, inputs, pooling_vec, sen_len, scope_name, is_reuse=False, type_=''):

        hiddens = self.dynamic_rnn(inputs, sen_len, self.max_sentence_len, scope_name, is_reuse, 'all')

        batch_size = tf.shape(inputs)[0]
        pooling_vec = tf.reshape(pooling_vec, [-1, 1, self.embedding_dim])
        pooling_vec = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * pooling_vec
        h_t = tf.reshape(tf.concat([hiddens, pooling_vec], 2), [-1, self.n_hidden + self.embedding_dim])
        #M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
        M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W_att) + self.b_att), self.U)
        alpha = self.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), sen_len, self.max_sentence_len)
        #self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])

        att_vecs = tf.reshape(tf.matmul(alpha, hiddens), [-1,  self.n_hidden])
        #index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        #hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        #h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))

        return att_vecs # batch_size X 2*n_hidden
        #return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], 1.0)

       
    def IAN(self, target, context, target_len, sen_len, keep_rate):
        """
        :params: self.target, self.x, self.x_fw, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        target_pooling = self.reduce_mean(tf.nn.embedding_lookup(self.embedding, target), target_len)
        context_pooling = self.reduce_mean(tf.nn.embedding_lookup(self.embedding, context), sen_len)
        context = tf.nn.embedding_lookup(self.embedding, context)
        target = tf.nn.embedding_lookup(self.embedding, target)
        context = tf.nn.dropout(context, keep_prob=keep_rate)
        target = tf.nn.dropout(target, keep_prob=keep_rate)
        with tf.name_scope('context_attention'):
            context = self.AT(context, target_pooling, sen_len, 'context', is_reuse=False) 
        with tf.name_scope('target_attention'):
            target = self.AT(target, context_pooling, target_len, 'target', is_reuse=False) 
        
        outputs = tf.concat([target, context], axis=1)
        
        logits = tf.matmul(outputs, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return logits

    def reduce_mean(self, inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs


    def forward(self, xa_inputs, hyper_inputs):
        with tf.name_scope('forward'):
            inputs_fw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_fw'])
            inputs_bw = tf.nn.embedding_lookup(self.embedding, xa_inputs['x_bw'])
            inputs_x = tf.nn.embedding_lookup(self.embedding, xa_inputs['x'])
            #target = self.reduce_mean(tf.nn.embedding_lookup(self.embedding, xa_inputs['target_words']), xa_inputs['target_len'])
            #batch_size = tf.shape(inputs_bw)[0]
            #target_expanded = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim]) + target
            #inputs_fw = tf.concat([inputs_fw, target], 2)
            #inputs_bw = tf.concat([inputs_bw, target], 2)
            logits = self.IAN(xa_inputs['target_words'], xa_inputs['x'], xa_inputs['target_len'], xa_inputs['sen_len'], hyper_inputs['keep_rate'])
        return logits

    def get_loss(self, logits, y_inputs, pri_prob_y):
        y = tf.argmax(y_inputs['y'], axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        pri_loss = tf.log(tf.gather(pri_prob_y, y))
        correct_pred = tf.equal(tf.argmax(logits, axis=1), y)
        acc = tf.reduce_mean(tf.to_float(correct_pred))
        return loss, acc, pri_loss

    def run(self, sess, train_data, test_data, n_iter, keep_rate, save_dir):
        self.init_global_step()
        input_placeholders = self.create_placeholders('xa')
        hyper_inputs = self.create_placeholders('hyper')
        print(input_placeholders, hyper_inputs)
        logits = self.forward(input_placeholders, hyper_inputs)
        y = tf.placeholder(tf.int32, [None, self.n_class], 'y')

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #cost = cost +  tf.reduce_mean(reg_losses)
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
        _dir = save_dir + '/logs/' + str(timestamp) + '_' +  '_r' + str(self.learning_rate) + '_b'  + '_l' + str(self.l2_reg)
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
                feed_data.update({'keep_rate': keep_rate})
                input_placeholders.update(hyper_inputs)
                feed_dict = self.get_feed_dict(input_placeholders, feed_data)
                feed_dict.update({y: get_y(samples)})
                
                _, step, summary = sess.run([optimizer, self.global_step, train_summary_op], feed_dict=feed_dict)
                train_summary_writer.add_summary(summary, step)
            acc, loss, cnt = 0., 0., 0
            for samples, in test_data:
                feed_data = self.prepare_data(samples)
                feed_data.update({'keep_rate': 1.0})
                input_placeholders.update(hyper_inputs)
                feed_dict = self.get_feed_dict(input_placeholders, feed_data)
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

    def prepare_data(self, samples): # , sentence_len, type_='', encoding='utf8'):
        sentence_len = self.max_sentence_len
        type_='IAN'
        encoding='utf8'
        word_to_id = self.word2idx
    
        x, y, sen_len = [], [], []
        x_r, sen_len_r = [], []
        x_l, sen_len_l = [], [] 
        target_word_list, target_len = [], []
        y_dict = {'positive': [1,0,0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}
        for sample in samples:
            target_words = [sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O']
            target_words = list(map(lambda w: word_to_id.get(w, 0), target_words))
            target_word_list.append(target_words + [0] * (sentence_len - len(target_words))) #?????
            target_len.append(len(target_words)) 
            #print(target_words) 
            if 'polarity' in sample:
                polarity = sample['polarity']
                y.append(y_dict[polarity])
            words = sample['tokens']
            words = [sample['tokens'][i] if sample['tags'][i] == 'O' else '$t$' for i in range(len(words))]
            words_l, words_r, words_all = [], [], []
            flag = True
            for word in words:
                if word == '$t$':
                    flag = False
                    continue
                if flag:
                    #if word in word_to_id:
                        #words_l.append(word_to_id[word])
                    words_l.append(word_to_id.get(word, word_to_id[UNK_TOKEN]))
                else:
                    #if word in word_to_id:
                        #words_r.append(word_to_id[word])
                    words_r.append(word_to_id.get(word, word_to_id[UNK_TOKEN]))
            if type_ == 'IAN':
                #words_l.extend(target_word)
                sen_len_l.append(len(words_l))
                x_l.append(words_l + [0] * (sentence_len - len(words_l)))
                tmp = words_r
                #tmp.reverse()
                sen_len_r.append(len(words_r))
                x_r.append(words_r + [0] * (sentence_len - len(words_r)))
                words_all = words_l + target_words + words_r
                #print(words_all)
                sen_len.append(len(words_all))
                x.append(words_all + [0] * (sentence_len - len(words_all)))
        return {'x': np.asarray(x),
                'sen_len': np.asarray(sen_len),
                'x_fw': np.asarray(x_l), 
                'sen_len_fw': np.asarray(sen_len_l), 
                'x_bw': np.asarray(x_r),
                'sen_len_bw':np.asarray(sen_len_r), 
                'target_words': np.asarray(target_word_list),
                'target_len': np.asarray(target_len),
                'y': np.asarray(y),
                }
    
def main(_):
    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open('../../../../data/se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
    test = pkl.load(open('../../../../data/se2014task06/tabsa-rest/test.pkl', 'rb'), encoding='latin')
    
    fns = ['../../../../data/se2014task06/tabsa-rest/train.pkl',
            '../../../../data/se2014task06/tabsa-rest/dev.pkl',
            '../../../../data/se2014task06/tabsa-rest/test.pkl',]

    data_dir = '../unlabel10k'
    #data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, embedding = preprocess_data(fns, '/Users/wdxu/data/glove/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()

        model = IANClassifier(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding,
                grad_clip=FLAGS.grad_clip)
	
        model.run(sess, train_it, test_it, FLAGS.n_iter, FLAGS.keep_rate, data_dir)

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.set_random_seed(1234)
    np.random.seed(1234)
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
    
    tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/train.raw', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/validate.raw', 'validating file')
    tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/test.raw', 'testing file')
    #tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_word_embedding_partial_100.txt', 'embedding file')
    #tf.app.flags.DEFINE_string('word_id_file_path', 'data/twitter/word_id.txt', 'word-id mapping file')
    tf.app.flags.DEFINE_string('type', 'TC', 'model type: ''(default), TD or TC')
    tf.app.flags.DEFINE_float('keep_rate', 0.5, 'keep rate')
    tf.app.flags.DEFINE_float('grad_clip', 5.0, 'gradient_clip')

    tf.app.run()
