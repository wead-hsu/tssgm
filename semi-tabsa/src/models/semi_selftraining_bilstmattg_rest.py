import sys
sys.path.append('../../')

import tensorflow as tf
from src.models.base_model import BaseModel
from src.models.classifier.tc_classifier import TCClassifier
from src.models.classifier.mem_classifier import MEMClassifier
from src.models.classifier.bilstm_att_g import BilstmAttGClassifier
from src.models.encoder.tc_encoder import TCEncoder
from src.models.decoder.tc_decoder import TCDecoder
from src.io.exp_logger import ExpLogger
from src.io.batch_iterator import BatchIterator
import pickle as pkl
import numpy as np
import os
import logging
from collections import Counter
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNK_TOKEN = "$unk$"
ASP_TOKEN = "$t$"


def preprocess_data(fns, pretrain_fn, data_dir, FLAGS):
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
    
    if os.path.exists(os.path.join(data_dir, 'vocab_sent.pkl')) and os.path.exists(os.path.join(data_dir, 'target_vocab_sent.pkl')):
        logger.info('Processed vocab already exists in {}'.format(data_dir))
        word2idx_sent = pkl.load(open(os.path.join(data_dir, 'vocab_sent.pkl'), 'rb'))
        target2idx_sent = pkl.load(open(os.path.join(data_dir, 'target_vocab_sent.pkl'), 'rb'))
    else:
        # keep the same format as in previous work
        words_sent = []
        target_sent = []
        if isinstance(fns, str): fns = [fns]
        for fn in fns:
            data          = pkl.load(open(fn, 'rb'), encoding='latin')
            if fn in [FLAGS.train_file_path, FLAGS.validate_file_path, FLAGS.test_file_path]:
                words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])] * 10000
            else:
                words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])]
            for sample in data:
                if fn in [FLAGS.train_file_path, FLAGS.validate_file_path, FLAGS.test_file_path]:
                    target_sent += [" ".join([sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O'])] * 10000
                else:
                    target_sent += [" ".join([sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O'])] 
               
        def build_vocab(words, tokens):
            words = Counter(words)
            word2idx = {token: i for i, token in enumerate(tokens)}
            word2idx.update({w[0]: i+len(tokens) for i, w in enumerate(words.most_common(20000))})
            return word2idx
        def build_target_vocab(targets, tokens):
            targets = Counter(targets)
            target2idx = {token: i for i, token in enumerate(tokens)}
            target2idx.update({w[0]: i+len(tokens) for i, w in enumerate(targets.most_common(20000))})
            return target2idx
        word2idx_sent = build_vocab(words_sent, [UNK_TOKEN, ASP_TOKEN])
        target2idx_sent = build_target_vocab(target_sent, [UNK_TOKEN, ASP_TOKEN])
        with open(os.path.join(data_dir, 'vocab_sent.pkl'), 'wb') as f:
            pkl.dump(word2idx_sent, f)
        logger.info('Vocabuary for input words has been created. shape={}'.format(len(word2idx_sent)))
    
        with open(os.path.join(data_dir, 'target_vocab_sent.pkl'), 'wb') as f:
            pkl.dump(target2idx_sent, f)
        logger.info('Target Vocabuary for input words has been created. shape={}'.format(len(target2idx_sent)))
    
    # create embedding from pretrained vectors
    if os.path.exists(os.path.join(data_dir, 'emb_sent.pkl')):
        logger.info('word embedding matrix already exisits in {}'.format(data_dir))
        emb_init_sent = pkl.load(open(os.path.join(data_dir, 'emb_sent.pkl'), 'rb'))
    else:
        if pretrain_fn is None:
            logger.info('Pretrained vector is not given, the embedding matrix is not created')
        else:
            pretrained_vectors = {str(l.split(" ")[0]): [float(n) for n in l.split(" ")[1:]] for l in open(pretrain_fn).readlines()}
            dim_emb = len(pretrained_vectors[list(pretrained_vectors.keys())[0]])
            def build_emb(pretrained_vectors, word2idx):
                emb_init = np.random.randn(len(word2idx), dim_emb) * 1e-2
                for w in word2idx:
                    if w in pretrained_vectors:
                        emb_init[word2idx[w]] = pretrained_vectors[w]
                    #else:
                    #    print(w)
                return emb_init
            emb_init_sent = build_emb(pretrained_vectors, word2idx_sent).astype('float32')
            with open(os.path.join(data_dir, 'emb_sent.pkl'), 'wb') as f:
                pkl.dump(emb_init_sent, f)
            logger.info('Pretrained vectors has been created from {}'.format(pretrain_fn))

    # create target embedding from pretrained vectors
    if os.path.exists(os.path.join(data_dir, 'target_emb_sent.pkl')):
        logger.info('target embedding matrix already exisits in {}'.format(data_dir))
        target_emb_init_sent = pkl.load(open(os.path.join(data_dir, 'target_emb_sent.pkl'), 'rb'))
    else:
        target_emb_init_sent = np.zeros([len(target2idx_sent), dim_emb], dtype = float)
        for target in target2idx_sent:
            for word in target.split():
                #if word2idx_sent[word] in emb_init_sent:
                if word in word2idx_sent:
                    target_emb_init_sent[target2idx_sent[target]] += emb_init_sent[word2idx_sent[word]]
                #else:
                #    print(word2idx_sent[word])
            target_emb_init_sent[target2idx_sent[target]] /= max(1, len(target.split()))
        with open(os.path.join(data_dir, 'target_emb_sent.pkl'), 'wb') as f:
            pkl.dump(target_emb_init_sent, f)
            logger.info('target pretrained vectors has been created from {}'.format(pretrain_fn))
    return word2idx_sent, target2idx_sent, emb_init_sent, target_emb_init_sent

def selftraining(sess, classifier, label_data, unlabel_data, test_data, FLAGS):
    xa_inputs = classifier.create_placeholders('xa')
    hyper_inputs = classifier.create_placeholders('hyper')
    y_inputs = classifier.create_placeholders('y')

    logits = classifier.forward(xa_inputs, hyper_inputs)
    loss, acc, _ = classifier.get_loss(logits, y_inputs, [0.0] * classifier.n_class)
    pred = tf.argmax(logits, axis=1)
    prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)

    import time, datetime
    timestamp = str(int(time.time()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_dir = FLAGS.save_dir + '/selftraining/' + str(timestamp) + '/' + __file__.split('.')[0]
    print(save_dir)
    logger = ExpLogger('semi_tabsa', save_dir)
    logger.write_args(vars(FLAGS)['__flags'])
    logger.write_variables(tf.trainable_variables())
    logger.file_copy(['*.py', 'encoder/*.py', 'decoder/*.py', 'classifier/*.py'])

    def get_feed_dict_help(classifier, plhs, data_dict, keep_rate, is_training):
        plh_dict = {}
        for plh in plhs: plh_dict.update(plh)
        data_dict.update({'keep_rate': keep_rate})
        data_dict.update({'is_training': is_training})
        feed_dict = classifier.get_feed_dict(plh_dict, data_dict)
        return feed_dict
    
    with tf.name_scope('train'):
        loss = tf.reduce_mean(loss)
        optimizer = classifier.training_op(loss, tf.trainable_variables(), FLAGS.grad_clip, 20, FLAGS.learning_rate, grads=None, opt='Adam')

    NUM_SELECT = 1000
    NUM_ITER = 500
    best_acc_in_rounds, best_f1_in_rounds = [], []
    while len(unlabel_data):
        tf.global_variables_initializer().run()
        test_it = BatchIterator(len(test_data), FLAGS.batch_size, [test_data], testing=True)
        print(len(unlabel_data))

        selected = []
        new_unlabel = []
        it_cnt = 0
        best_acc, best_f1 = 0, 0
        while True:

            train_it = BatchIterator(len(label_data), FLAGS.batch_size, [label_data], testing=False)
            unlabel_it = BatchIterator(len(unlabel_data), FLAGS.batch_size, [unlabel_data], testing=True)
            
            for samples, in train_it:
                it_cnt += 1
                if it_cnt > NUM_ITER:
                    break
                feed_dict = get_feed_dict_help(classifier,
                        plhs=[xa_inputs, y_inputs, hyper_inputs],
                        data_dict=classifier.prepare_data(samples),
                        keep_rate=FLAGS.keep_rate,
                        is_training=True)

                _, _loss, _acc, _step = sess.run([optimizer, loss, acc, classifier.global_step], feed_dict=feed_dict)
                #print('Train: step {}, acc {}, loss {}'.format(it_cnt, _acc, _loss))
                    
                ### proc test
                test_acc, cnt = 0, 0
                y_true = []
                y_pred = []
                for samples, in test_it:
                    data_dict = classifier.prepare_data(samples)
                    feed_dict = get_feed_dict_help(classifier,
                            plhs=[xa_inputs, y_inputs, hyper_inputs],
                            data_dict=data_dict,
                            keep_rate=1.0,
                            is_training=False)
                
                    num = len(samples)
                    _acc, _loss, _pred, _step = sess.run([acc, loss, pred, classifier.global_step], feed_dict=feed_dict)
                    y_pred.extend(list(_pred))
                    y_true.extend(list(np.argmax(data_dict['y'],1)))
                    test_acc += _acc * num
                    cnt += num
                test_acc = test_acc / cnt
                test_f1 = f1_score(y_true, y_pred, average='macro')
                logger.info('Test: step {}, test acc={:.6f}, test f1={:.6f}'.format(it_cnt, test_acc, test_f1))
                best_f1 = max(best_f1, test_f1)
                
                ### proc unlabel
                if best_acc < test_acc:
                    best_acc = test_acc
                    _unlabel = []
                    _preds = []
                    _probs = []
                    y_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}

                    for samples, in unlabel_it:
                        feed_dict = get_feed_dict_help(classifier,
                                plhs=[xa_inputs, hyper_inputs],
                                data_dict=classifier.prepare_data(samples),
                                keep_rate=1.0,
                                is_training=False)
                
                        _pred, _prob = sess.run([pred, prob], feed_dict=feed_dict)
                        _unlabel.extend(samples)
                        _preds.extend(list(_pred))
                        _probs.extend(list(_prob))

                    top_k_id = np.argsort(_probs)[::-1][:NUM_SELECT]
                    remain_id = np.argsort(_probs)[::-1][NUM_SELECT:]
                    selected = [_unlabel[idx] for idx in top_k_id]
                    preds = [_preds[idx] for idx in top_k_id]
                    for idx, sample in enumerate(selected):
                        sample['polarity'] = y_dict[preds[idx]]
                    new_unlabel = [_unlabel[idx] for idx in remain_id]

            if it_cnt > NUM_ITER:
                best_acc_in_rounds.append(best_acc)
                best_f1_in_rounds.append(best_f1)
                logger.info(str(best_acc_in_rounds) + str(best_f1_in_rounds))
                break

        label_data.extend(selected)
        unlabel_data = new_unlabel

    #print(max(best_acc_in_rounds), max(best_f1_in_rounds))
    logger.info(str(best_acc_in_rounds) + str(best_f1_in_rounds))
    
def main(_):
    FLAGS = tf.app.flags.FLAGS

    import time, datetime
    timestamp = str(int(time.time()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = FLAGS.save_dir + '/selftraining/'

    train = pkl.load(open(FLAGS.train_file_path, 'rb'), encoding='latin')
    unlabel = pkl.load(open(FLAGS.unlabel_file_path, 'rb'), encoding='latin')[:FLAGS.n_unlabel]
    test = pkl.load(open(FLAGS.test_file_path, 'rb'), encoding='latin')
    val = pkl.load(open(FLAGS.validate_file_path, 'rb'), encoding='latin')

    fns = [FLAGS.train_file_path,  FLAGS.test_file_path, FLAGS.unlabel_file_path]
    #data_dir = 'classifier/data/rest/bilstmattg-cbow/'
    data_dir = 'classifier/data/rest/bilstmattg/'
    emb_file = "../../../data/glove.6B/glove.6B.300d.txt"
    #emb_file = "../../../data/glove.840B/glove.840B.300d.txt"
    #emb_file = "../../../data/se2014task06//tabsa-rest/cbow.unlabel.300d.txt"
    word2idx, target2idx, word_embedding, target_embedding = preprocess_data(fns, emb_file, data_dir, FLAGS)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()
        
        """
        if word_embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(word2idx), embedding_dim]) * 1e-2
            word_embedding = tf.get_variable('word_embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(wemb_init))
        elif isinstance(word_embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(word_embedding.shape)))
            word_embedding = tf.constant(word_embedding, name='embedding')
            #self.word_embedding = tf.get_variable('word_embedding', [len(word2idx), embedding_dim], initializer=tf.constant_initializer(word_embedding))
        elif isinstance(word_embedding, tf.Tensor):
            logger.info('Import tensor as the embedding: '.format(word_embedding.name))
            word_embedding = word_embedding
        else:
            raise Exception('Embedding type {} is not supported'.format(type(word_embedding)))

        if target_embedding is None:
            logger.info('No embedding is given, initialized randomly')
            wemb_init = np.random.randn([len(target2idx), embedding_dim]) * 1e-2
            target_embedding = tf.get_variable('target_embedding', [len(target2idx), embedding_dim], initializer=tf.constant_initializer(wemb_init))
        elif isinstance(target_embedding, np.ndarray):
            logger.info('Numerical embedding is given with shape {}'.format(str(target_embedding.shape)))
            target_embedding = tf.constant(target_embedding, name='embedding')
#            self.target_embedding = tf.get_variable('target_embedding', [len(target2idx), embedding_dim], initializer=tf.constant_initializer(target_embedding))
        elif isinstance(target_embedding, tf.Tensor):
            logger.info('Import tensor as the embedding: '.format(target_embedding.name))
            target_embedding = target_embedding
        else:
            raise Exception('Embedding type {} is not supported'.format(type(embedding)))
        """

        #TODO: Take the network graph building codes to a new module. 
        #self.classifier = self.create_classifier(self.classifier_type)

        classifier = BilstmAttGClassifier(word2idx=word2idx, 
                     embedding_dim=FLAGS.embedding_dim, 
                     n_hidden=FLAGS.n_hidden, 
                     learning_rate=FLAGS.learning_rate, 
                     n_class=FLAGS.n_class, 
                     max_sentence_len=FLAGS.max_sentence_len, 
                     l2_reg=FLAGS.l2_reg, 
                     embedding=word_embedding,
                     grad_clip=FLAGS.grad_clip)

        selftraining(sess, classifier, train, unlabel, test, FLAGS)

    return

if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
    tf.app.flags.DEFINE_integer('n_hidden_ae', 100, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 95, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.01, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
    tf.app.flags.DEFINE_integer('n_unlabel', 5000, 'number of unlabeled')
    
    tf.app.flags.DEFINE_string('train_file_path', '../../../data/se2014task06/tabsa-rest/train.pkl', 'training file')
    tf.app.flags.DEFINE_string('unlabel_file_path', '../../../data/se2014task06/tabsa-rest/unlabel.clean.pkl', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', '../../../data/se2014task06/tabsa-rest/dev.pkl', 'training file')
    tf.app.flags.DEFINE_string('test_file_path', '../../../data/se2014task06/tabsa-rest/test.pkl', 'training file')
    tf.app.flags.DEFINE_string('classifier_type', 'BILSTM_ATT_G', 'model type: ''(default), MEM or TD or TC or BILSTM_ATT_G')
    tf.app.flags.DEFINE_float('keep_rate', 0.5, 'keep rate')
    tf.app.flags.DEFINE_string('decoder_type', 'fclstm', '[sclstm, lstm]')
    tf.app.flags.DEFINE_float('grad_clip', 100, 'gradient_clip, <0 == None')
    tf.app.flags.DEFINE_integer('dim_z', 50, 'dimension of z latent variable')
    tf.app.flags.DEFINE_float('alpha', 5.0, 'weight of alpha')
    tf.app.flags.DEFINE_string('save_dir', 'logs', 'directory of save file')

    tf.app.run()
