import tensorflow as tf
from src.models.base_model import BaseModel
from src.models.classifier.tc_classifier import TCClassifier
from src.models.encoder.tc_encoder import TCEncoder
from src.models.decoder.tc_decoder import TCDecoder

class SemiTABSA(BaseModel):
    def __init__(self, word2idx, embedding_dim, batch_size, n_hidden, learning_rate, n_class, max_sentence_len, l2_reg, embedding, dim_z, pri_prob_y):
        super(self, SemiTABSA).__init__(self)

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.word2idx = word2idx
        self.dim_z = dim_z

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


        self.classifier = TCClassifier(word2idx=word2idx, 
                    embedding_dim=embedding_dim, 
                    n_hidden=n_hidden, 
                    learning_rate=learning_rate, 
                    n_class=n_class, 
                    max_sentence_len=max_sentence_len, 
                    l2_reg=l2_reg, 
                    embedding=self.embedding,
                    dim_z=dim_z)

        self.encoder = TCEncoder(word2idx=word2idx, 
                    embedding_dim=embedding_dim, 
                    n_hidden=n_hidden, 
                    learning_rate=learning_rate, 
                    n_class=n_class, 
                    max_sentence_len=max_sentence_len, 
                    l2_reg=l2_reg, 
                    embedding=self.embedding,
                    dim_z=dim_z)

        self.decoder = TCDecoder(word2idx=word2idx, 
                    embedding_dim=embedding_dim, 
                    n_hidden=n_hidden, 
                    learning_rate=learning_rate, 
                    n_class=n_class, 
                    max_sentence_len=max_sentence_len, 
                    l2_reg=l2_reg, 
                    embedding=self.embedding,
                    dim_z=dim_z)

        self.klw = tf.get_variable(tf.float32)
        self.pri_prob_y =  pri_prob_y # distribution of y in the dataset
    
    def run(self, sess, train_l, train_u, n_iter, keep_rate):
        with tf.variable_scope('labeled'):
            with tf.variable_scope('classifier'):
                self.classifier_xa_l = self.classifier.create_placeholders('xa')
                self.classifier_y_l = self.classifier.create_placeholders('y')
                logits_l = classifier.forward(self.classifier_x_l)
                classifier_loss_l = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.classifier_y_l,
                        logits=logits)
                pri_loss_l = tf.log(self.pri_prob_y[tf.argmax(self.classifier_y_l, 1)])

            with tf.variable_scope('encoder'):
                self.encoder_xa_l = self.encoder.create_placeholders('xa')
                self.encoder_y_l = self.encoder.create_placeholders('y')
                z_pst, z_pri, encoder_loss_l = self.encoder.forward(self.encoder_xa_l, self.encoder_y_l)

            with tf.variable_scope('decoder'):
                self.decoder_xa_l = self.decoder.create_placeholders('xa') #x is included since x is generated sequentially
                self.decoder_y_l = self.decoder.createa_placeholders('y')
                decoder_loss_l = self.decoder.forward(self.decoder_xa_l, self.y_l, z_pst)
            elbo_l = encoder_loss_l * self.klw + decoder_loss_l - pri_loss_l

        self.loss_l = elbo_l
        self.loss_c = classifier_loss_l
        """
        with tf.variable_scope('unlabeled'):
            with tf.variable_scope('classifier'):
                self.classifier_xa_u = self.classifier.create_placeholders('xa')
                logits_u = classifier.forward(self.classifier_inputs_u)
                predict_u = tf.softmax(logits_u)
                classifier_loss_u = tf.losses.softmax_cross_entropy(predict_u, predict_u)

                self.recons_loss_u, self.kl_loss_u, self.loss_sum_u = [], [], []
                encoder_loss_u, decoder_loss_u = [], []
                
            elbo_u = []
            self.encoder_xa_u = self.encoder.create_placeholders('xa')
            self.decoder_xa_u = self.decoder.create_placeholders('xa')
            for idx in range(self.n_class):
                with tf.variable_scope('encoder'):
                    _label = tf.constant(idx)
                    _label = tf.tile([idx], [tf.shape(self.classifier_xa_u[0])[0]])
                    _z_pst, _, _encoder_loss = self.encoder.forward(self.encoder_xa_u)
                    encoder_loss_u.append(_encoder_loss * self.klw)
                    _pri_loss_u = tf.log(self.pri_prob_y[idx])
            
                with tf.variable_scope('decoedr'):
                    _decoder_loss = self.decoder.forward(self.decoder_xa_u, _label, _z_pst)
                    decoder_loss_u.append(_decoder_loss)

                _elbo_u = _encoder_loss_u * self.klw + _decoder_loss_u - _pri_loss_u
                elbo_u.append(_elbo_u)

        self.loss_u = tf.add_n([(elbo_u[idx] * predict_u[:, idx] for idx in range(self.n_class)])
        """

        self.loss = elbo_l + classifier_loss_l # for test
    
        
def main(_):
    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open('../../../../data/se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
    test = pkl.load(open('../../../../data/se2014task06/tabsa-rest/test.pkl', 'rb'), encoding='latin')
    
    fns = ['../../../../data/se2014task06/tabsa-rest/train.pkl',
            '../../../../data/se2014task06/tabsa-rest/dev.pkl',
            '../../../../data/se2014task06/tabsa-rest/test.pkl',]

    data_dir = 'classifier/0617'
    #data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, embedding = preprocess_data(fns, '/Users/wdxu/data/glove/glove.6B/glove.6B.300d.txt', data_dir)
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)

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
                dim_z = 3)


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

    tf.app.run()
