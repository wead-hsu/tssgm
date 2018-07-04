import sys
sys.path.append('../../')

import tensorflow as tf
from src.models.base_model import BaseModel
from src.models.classifier.tc_classifier import TCClassifier
#from src.models.classifier.mem_classifier import MEMClassifier
from src.models.encoder.tc_encoder import TCEncoder
from src.models.decoder.tc_decoder import TCDecoder
from src.io.exp_logger import ExpLogger
import pickle as pkl
import numpy as np
import os
import logging
from collections import Counter
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNK_TOKEN = "$unk$"
ASP_TOKEN = "$t$"

def get_batch(dataset):
    """ to get batch from an iterator, whenever the ending is reached. """
    while True:
        try:
            batch = dataset.next()
            break
        except:
            pass
    return batch

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
        pretrained_vectors = {str(l.split()[0]): [float(n) for n in l.split()[1:]] for l in open(pretrain_fn).readlines()}
        #print(list(pretrained_vectors.keys())[:100])
        words_sent = []
        if isinstance(fns, str): fns = [fns]
        for fn in fns:
            data          = pkl.load(open(fn, 'rb'), encoding='latin')
            words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])]

        def build_vocab(words, tokens):
            words = Counter(words)
            word2idx = {token: i for i, token in enumerate(tokens)}
            for w in words.most_common(20000):
                if w[0] in pretrained_vectors:
                    word2idx[w[0]] = len(word2idx)
            #word2idx.update({w[0]: i+len(tokens) for i, w in enumerate(words.most_common(20000)) if w[0] in pretrained_vectors})
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

class SemiTABSA(BaseModel):
    def __init__(self, word2idx, embedding_dim, batch_size, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, dim_z, pri_prob_y, decoder_type, grad_clip, n_hidden_ae, position_enc, bidirection_enc, position_dec, bidirection_dec, classifier_type):
        super(SemiTABSA, self).__init__()

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.word2idx = word2idx
        self.dim_z = dim_z
        self.decoder_type = decoder_type
        self.classifier_type = classifier_type
        self.grad_clip = grad_clip
        self.n_hidden_ae = n_hidden_ae
        self.pri_prob_y = tf.Variable(pri_prob_y, trainable=False)
        self.position_enc = position_enc
        self.bidirection_enc = bidirection_enc
        self.position_dec = position_dec
        self.bidirection_dec = bidirection_dec

        if embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(wemb_init))
        elif isinstance(embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(embedding.shape)))
            #self.embedding = tf.constant(embedding, name='embedding')
            self.embedding = tf.get_variable('embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(embedding), trainable=False)
        elif isinstance(embedding, tf.Tensor) or isinstance(embedding, tf.Variable):
            logger.info('Import tensor as the embedding: '.format(embedding.name))
            self.embedding = embedding
        else:
            raise Exception('Embedding type {} is not supported'.format(type(embedding)))

        #TODO: Take the network graph building codes to a new module. 
        #self.classifier = self.create_classifier(self.classifier_type)
        with tf.variable_scope('classifier'):
            if self.classifier_type == "TC":
                self.classifier = TCClassifier(word2idx=word2idx, 
                        embedding_dim=embedding_dim, 
                        n_hidden=n_hidden, 
                        learning_rate=learning_rate, 
                        n_class=n_class, 
                        max_sentence_len=max_sentence_len, 
                        l2_reg=l2_reg, 
                        embedding=self.embedding,
                        grad_clip=self.grad_clip,
                        )
            elif self.classifier_type == "TD":
                pass
            elif self.classifier_type == "MEM":
                #TODO: Add hyper-params Config.py
                self.classifier = MEMClassifier()
        
        with tf.variable_scope('encoder'):
            self.encoder = TCEncoder(word2idx=word2idx, 
                    embedding_dim=embedding_dim, 
                    n_hidden=n_hidden_ae, 
                    learning_rate=learning_rate, 
                    n_class=n_class, 
                    max_sentence_len=max_sentence_len, 
                    l2_reg=l2_reg, 
                    embedding=self.embedding,
                    dim_z=dim_z,
                    grad_clip=self.grad_clip,
                    position=self.position_enc,
                    bidirection=self.bidirection_enc,
                    )
        
        with tf.variable_scope('decoder'):
            self.decoder = TCDecoder(word2idx=word2idx, 
                    embedding_dim=embedding_dim, 
                    n_hidden=n_hidden_ae, 
                    learning_rate=learning_rate, 
                    n_class=n_class, 
                    max_sentence_len=max_sentence_len, 
                    l2_reg=l2_reg, 
                    embedding=self.embedding,
                    dim_z=dim_z,
                    decoder_type=self.decoder_type,
                    grad_clip=self.grad_clip,
                    position=self.position_dec,
                    bidirection=self.bidirection_dec,
                    )

        self.klw = tf.placeholder(tf.float32, [], 'klw')
    
    def run(self, sess, train_data_l, train_data_u, test_data, n_iter, keep_rate, save_dir, batch_size, alpha, FLAGS):
        self.init_global_step()
        with tf.name_scope('labeled'):
            with tf.variable_scope('classifier'):
                self.classifier_xa_l = self.classifier.create_placeholders('xa')
                self.classifier_y_l = self.classifier.create_placeholders('y')
                self.classifier_hyper_l = self.classifier.create_placeholders('hyper')
                logits_l = self.classifier.forward(self.classifier_xa_l, self.classifier_hyper_l)
                classifier_loss_l, classifier_acc_l, pri_loss_l = self.classifier.get_loss(logits_l, self.classifier_y_l, self.pri_prob_y)

            with tf.variable_scope('encoder'):
                self.encoder_xa_l = self.encoder.create_placeholders('xa')
                self.encoder_y_l = self.encoder.create_placeholders('y')
                self.encoder_hyper_l = self.encoder.create_placeholders('hyper')
                z_pst, z_pri, encoder_loss_l = self.encoder.forward(self.encoder_xa_l, self.encoder_y_l, self.encoder_hyper_l)

            with tf.variable_scope('decoder'):
                self.decoder_xa_l = self.decoder.create_placeholders('xa') #x is included since x is generated sequentially
                self.decoder_y_l = self.decoder.create_placeholders('y')
                self.decoder_hyper_l = self.decoder.create_placeholders('hyper')
                decoder_loss_l, ppl_fw_l, ppl_bw_l, ppl_l = self.decoder.forward(self.decoder_xa_l, self.decoder_y_l, z_pst, self.decoder_hyper_l)
            elbo_l = encoder_loss_l * self.klw + decoder_loss_l - pri_loss_l

        self.loss_l = elbo_l
        self.loss_c = classifier_loss_l
        
        with tf.name_scope('unlabeled'):
            with tf.variable_scope('classifier', reuse=True):
                self.classifier_xa_u = self.classifier.create_placeholders('xa')
                self.classifier_hyper_u = self.classifier.create_placeholders('hyper')
                logits_u = self.classifier.forward(self.classifier_xa_u, self.classifier_hyper_u)
                predict_u = tf.nn.softmax(logits_u)
                classifier_entropy_u = tf.losses.softmax_cross_entropy(predict_u, predict_u)

            encoder_loss_u, decoder_loss_u = [], []
            elbo_u = []
            self.encoder_xa_u = self.encoder.create_placeholders('xa')
            self.encoder_hyper_u = self.encoder.create_placeholders('hyper')
            self.decoder_xa_u = self.decoder.create_placeholders('xa')
            self.decoder_hyper_u = self.decoder.create_placeholders('hyper')
            batch_size = tf.shape(list(self.encoder_xa_u.values())[0])[0]
            for idx in range(self.n_class):
                with tf.variable_scope('encoder', reuse=True):
                    _label = tf.gather(tf.eye(self.n_class), idx)
                    _label = tf.tile(_label[None, :], [batch_size, 1])
                    _z_pst, _, _encoder_loss = self.encoder.forward(self.encoder_xa_u, {'y':_label}, self.encoder_hyper_u)
                    encoder_loss_u.append(_encoder_loss * self.klw)
                    _pri_loss_u = tf.log(tf.gather(self.pri_prob_y, idx))
            
                with tf.variable_scope('decoder', reuse=True):
                    _decoder_loss, _, _, _ = self.decoder.forward(self.decoder_xa_u, {'y':_label}, _z_pst, self.decoder_hyper_u)
                    decoder_loss_u.append(_decoder_loss)

                _elbo_u = _encoder_loss * self.klw + _decoder_loss# - _pri_loss_u
                elbo_u.append(_elbo_u)

        self.loss_u = tf.add_n([elbo_u[idx] * predict_u[:, idx] for idx in range(self.n_class)]) + classifier_entropy_u
        self.loss = tf.reduce_mean(self.loss_l + classifier_loss_l * alpha + self.loss_u)
        batch_size_l = tf.shape(decoder_loss_l)[0]
        decoder_loss_l = tf.reduce_mean(decoder_loss_l)

        with tf.name_scope('train'):
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)
            optimizer = self.training_op(self.loss, tf.trainable_variables(), self.grad_clip, 20, self.learning_rate)
        
        summary_kl = tf.summary.scalar('kl', tf.reduce_mean(encoder_loss_l))
        summary_loss = tf.summary.scalar('loss', self.loss)
        summary_loss_l = tf.summary.scalar('loss_l', tf.reduce_mean(self.loss_l))
        summary_loss_u = tf.summary.scalar('loss_u', tf.reduce_mean(self.loss_u))
        summary_acc = tf.summary.scalar('acc', classifier_acc_l)
        summary_ppl_fw = tf.summary.scalar('ppl_fw', ppl_fw_l)
        summary_ppl_bw = tf.summary.scalar('ppl_bw', ppl_bw_l)
        summary_ppl = tf.summary.scalar('ppl', ppl_l)
        train_summary_op = tf.summary.merge_all()

        test_acc = tf.placeholder(tf.float32, [])
        test_ppl = tf.placeholder(tf.float32, [])
        summary_acc_test = tf.summary.scalar('test_acc', test_acc)
        summary_ppl_test = tf.summary.scalar('test_ppl', test_ppl)
        test_summary_op = tf.summary.merge([summary_acc_test, summary_ppl_test])

        logger = ExpLogger('semi_tabsa', save_dir)
        logger.write_args(FLAGS)
        logger.write_variables(tf.trainable_variables())
        logger.file_copy(['semi_tabsa.py', 'encoder/*.py', 'decoder/*.py', 'classifier/*.py'])

        summary_writer = tf.summary.FileWriter(save_dir + '/', sess.graph)
        #test_summary_writer = tf.summary.FileWriter(save_dir + '/', sess.graph)
        #validate_summary_writer = tf.summary.FileWriter(save_dir + '/validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        def get_batch(dataset):
            """ to get batch from an iterator, whenever the ending is reached. """
            while True:
                try:
                    batch = dataset.next()
                    break
                except:
                    pass
            return batch
        
        def get_feed_dict_help(plhs, data_dict, keep_rate, is_training):
            plh_dict = {}
            for plh in plhs: plh_dict.update(plh)
            data_dict.update({'keep_rate': keep_rate})
            data_dict.update({'is_training': is_training})
            feed_dict = self.get_feed_dict(plh_dict, data_dict)
            return feed_dict

        max_acc = 0.
        for i in range(n_iter):
            #for train, _ in self.get_batch_data(train_data, keep_rate):
            for samples, in train_data_l:
                feed_dict_clf_l = get_feed_dict_help(plhs=[self.classifier_xa_l, self.classifier_y_l, self.classifier_hyper_l],
                        data_dict=self.classifier.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)
                
                feed_dict_enc_l = get_feed_dict_help(plhs=[self.encoder_xa_l, self.encoder_y_l, self.encoder_hyper_l],
                        data_dict=self.encoder.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)
 
                feed_dict_dec_l = get_feed_dict_help(plhs=[self.decoder_xa_l, self.decoder_y_l, self.decoder_hyper_l],
                        data_dict=self.decoder.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)
                
                samples, = get_batch(train_data_u)
                feed_dict_clf_u = get_feed_dict_help(plhs=[self.classifier_xa_u, self.classifier_hyper_u],
                        data_dict=self.classifier.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)
                
                feed_dict_enc_u = get_feed_dict_help(plhs=[self.encoder_xa_u, self.encoder_hyper_u],
                        data_dict=self.encoder.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)
 
                feed_dict_dec_u = get_feed_dict_help(plhs=[self.decoder_xa_u, self.decoder_hyper_u],
                        data_dict=self.decoder.prepare_data(samples),
                        keep_rate=keep_rate,
                        is_training=True)

                feed_dict = {}
                feed_dict.update(feed_dict_clf_l)
                feed_dict.update(feed_dict_enc_l)
                feed_dict.update(feed_dict_dec_l)
                feed_dict.update(feed_dict_clf_u)
                feed_dict.update(feed_dict_enc_u)
                feed_dict.update(feed_dict_dec_u)
                feed_dict.update({self.klw: 0.0001})

                _, _acc, _loss, _ppl, _step, summary = sess.run([optimizer, classifier_acc_l, decoder_loss_l, ppl_l, self.global_step, train_summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, _step)
                if np.random.rand() < 1/4:
                    print(_acc, _loss, _ppl, _step)
            
            acc, ppl, loss, cnt = 0., 0., 0., 0
            for samples, in test_data:
                feed_dict_clf_l = get_feed_dict_help(plhs=[self.classifier_xa_l, self.classifier_y_l, self.classifier_hyper_l],
                        data_dict=self.classifier.prepare_data(samples),
                        keep_rate=1.0,
                        is_training=False)
                
                feed_dict_enc_l = get_feed_dict_help(plhs=[self.encoder_xa_l, self.encoder_y_l, self.encoder_hyper_l],
                        data_dict=self.encoder.prepare_data(samples),
                        keep_rate=1.0,
                        is_training=False)
 
                feed_dict_dec_l = get_feed_dict_help(plhs=[self.decoder_xa_l, self.decoder_y_l, self.decoder_hyper_l],
                        data_dict=self.decoder.prepare_data(samples),
                        keep_rate=1.0,
                        is_training=False)

                feed_dict = {}
                feed_dict.update(feed_dict_clf_l)
                feed_dict.update(feed_dict_enc_l)
                feed_dict.update(feed_dict_dec_l)
                feed_dict.update({self.klw: 0})

                num, _acc, _loss, _ppl, _step = sess.run([batch_size_l, classifier_acc_l, decoder_loss_l, ppl_l, self.global_step], feed_dict=feed_dict)
                acc += _acc * num
                ppl += _ppl * num
                loss += _loss * num
                cnt += num
            #print(cnt)
            #print(acc)
            summary, _step = sess.run([test_summary_op, self.global_step], feed_dict={test_acc: acc/cnt, test_ppl: ppl/cnt})
            summary_writer.add_summary(summary, _step)
            logger.info('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(_step, loss / cnt, acc / cnt))
            print(save_dir)
            if acc / cnt > max_acc:
                max_acc = acc / cnt

        logger.info('Optimization Finished! Max acc={}'.format(max_acc))

        logger.info('Learning_rate={}, iter_num={}, hidden_num={}, l2={}'.format(
            self.learning_rate,
            n_iter,
            self.n_hidden,
            self.l2_reg
        ))

def main(_):
    FLAGS = tf.app.flags.FLAGS
    FLAGS.bidirection_enc = True if FLAGS.bidirection_enc == 'True' else False
    FLAGS.bidirection_dec = True if FLAGS.bidirection_dec == 'True' else False
    FLAGS.sharefc         = True if FLAGS.sharefc         == 'True' else False

    import time, datetime
    timestamp = str(int(time.time()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = FLAGS.save_dir + '/' + str(timestamp) + '_' +  '_r' + str(FLAGS.learning_rate) + '_l2' + str(FLAGS.l2_reg)\
                + '_alpha' + str(FLAGS.alpha) + '_bs' + str(FLAGS.batch_size) + '_hidae' + str(FLAGS.n_hidden_ae)\
                + '_dz' + str(FLAGS.dim_z)  + '_dec' + str(FLAGS.decoder_type) + '_u' + str(FLAGS.n_unlabel)\
                + '_penc' + str(FLAGS.position_enc) + '_bienc' + str(FLAGS.bidirection_enc)\
                + '_pdec' + str(FLAGS.position_dec) + '_bidec' + str(FLAGS.bidirection_dec)\
                + '_sharefc' + str(FLAGS.sharefc) \
                + '_trunctdis'
                #+ '_vochas10kunl_addunkpd_noapconcattindec_kl1e-4_noh_filteremb_trunctdis_sharefc_decfcdepb'
    #save_dir = 'tmp'

    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open(FLAGS.train_file_path, 'rb'), encoding='latin')
    unlabel = pkl.load(open(FLAGS.unlabel_file_path, 'rb'), encoding='latin')[:FLAGS.n_unlabel]
    test = pkl.load(open(FLAGS.test_file_path, 'rb'), encoding='latin')

    def get_y(samples):
        y_dict = {'positive': [1,0,0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}
        ys = [y_dict[sample['polarity']] for sample in samples]
        return ys

    y = get_y(train)
    pri_prob_y = (np.sum(y, axis=0)/len(y)).astype('float32')
    #pri_prob_y = np.ones(FLAGS.n_class).astype('float32')/FLAGS.n_class
    print(pri_prob_y)
    
    fns = [FLAGS.train_file_path, FLAGS.unlabel_file_path, FLAGS.test_file_path]

    data_dir = 'unlabel10k_filter'
    word2idx, embedding = preprocess_data(fns, '../../../data/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    unlabel_it = BatchIterator(len(unlabel), FLAGS.batch_size, [unlabel], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=True)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()
        
        model = SemiTABSA(word2idx=word2idx, 
                embedding_dim=FLAGS.embedding_dim, 
                batch_size=FLAGS.batch_size, 
                n_hidden=FLAGS.n_hidden, 
                learning_rate=FLAGS.learning_rate, 
                n_class=FLAGS.n_class, 
                max_sentence_len=FLAGS.max_sentence_len, 
                l2_reg=FLAGS.l2_reg, 
                embedding=embedding,
                dim_z=FLAGS.dim_z,
                pri_prob_y=pri_prob_y,
                decoder_type=FLAGS.decoder_type,
                grad_clip=FLAGS.grad_clip,
                n_hidden_ae=FLAGS.n_hidden_ae,
                position_enc=FLAGS.position_enc,
                bidirection_enc=FLAGS.bidirection_enc,
                position_dec=FLAGS.position_dec,
                bidirection_dec=FLAGS.bidirection_dec,
                classifier_type=FLAGS.classifier_type,
                )

        model.run(sess, train_it, unlabel_it, test_it, FLAGS.n_iter, FLAGS.keep_rate, save_dir, FLAGS.batch_size, FLAGS.alpha, vars(FLAGS)['__flags'])

if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
    tf.app.flags.DEFINE_integer('n_hidden_ae', 100, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 85, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 50, 'number of train iter')
    tf.app.flags.DEFINE_integer('n_unlabel', 10000, 'number of unlabeled')
    
    tf.app.flags.DEFINE_string('train_file_path', '../../../data/se2014task06/tabsa-rest/train.pkl', 'training file')
    tf.app.flags.DEFINE_string('unlabel_file_path', '../../../data/se2014task06/tabsa-rest/unlabel.clean.pkl', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', '../../../data/se2014task06/tabsa-rest/dev.pkl', 'training file')
    tf.app.flags.DEFINE_string('test_file_path', '../../../data/se2014task06/tabsa-rest/test.pkl', 'training file')
    tf.app.flags.DEFINE_string('classifier_type', 'TC', 'model type: ''(default), TD or TC')
    tf.app.flags.DEFINE_float('keep_rate', 0.5, 'keep rate')
    tf.app.flags.DEFINE_string('decoder_type', 'sclstm', '[sclstm, lstm, gelstm, fclstm]')
    tf.app.flags.DEFINE_float('grad_clip', 5, 'gradient_clip, <0 == None')
    tf.app.flags.DEFINE_integer('dim_z', 50, 'dimension of z latent variable')
    tf.app.flags.DEFINE_float('alpha', 5.0, 'weight of alpha')
    tf.app.flags.DEFINE_string('save_dir', '.', 'directory of save file')
    tf.app.flags.DEFINE_string('position_enc', 'binary', '[binary, distance, ], None for no position embedding')
    tf.app.flags.DEFINE_string('bidirection_enc', 'True', 'boolean')
    tf.app.flags.DEFINE_string('position_dec', 'binary', '[binary, distance, ], None for no position embedding')
    tf.app.flags.DEFINE_string('bidirection_dec', 'True', 'boolean')
    tf.app.flags.DEFINE_string('sharefc', 'True', 'boolean')

    tf.app.run()
