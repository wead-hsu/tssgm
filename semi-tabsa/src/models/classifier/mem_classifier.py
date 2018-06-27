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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
UNK_TOKEN = "$unk$"
ASP_TOKEN = "$t$"
PAD_TOKEN = "$pad$"

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
            words_sent   += [w for sample in data for i, w in enumerate(sample['tokens'])]
            for sample in data:
                target_sent += [" ".join([sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O'])]
   
        def build_vocab(words, tokens):
            words = Counter(words)
            word2idx = {token: i for i, token in enumerate(tokens)}
            word2idx.update({w[0]: i+len(tokens) for i, w in enumerate(words.most_common())})
            return word2idx
        def build_target_vocab(targets, tokens):
            targets = Counter(targets)
            target2idx = {token: i for i, token in enumerate(tokens)}
            target2idx.update({w[0]: i+len(tokens) for i, w in enumerate(targets.most_common())})
            return target2idx
        word2idx_sent = build_vocab(words_sent, [UNK_TOKEN, ASP_TOKEN])
        target2idx_sent = build_target_vocab(target_sent, [UNK_TOKEN, ASP_TOKEN])
        with open(os.path.join(data_dir, 'vocab_sent.pkl'), 'wb') as f:
            pkl.dump(word2idx_sent, f)
        logger.info('Vocabuary for input words has been created. shape={}'.format(len(word2idx_sent)))
    
        with open(os.path.join(data_dir, 'target_vocab_sent.pkl'), 'wb') as f:
            pkl.dump(target2idx_sent, f)
        logger.info('Target Vocabuary for input words has been created. shape={}'.format(len(target2idx_sent)))
    
    dim_emb=300
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
                    #else:
                    #    print(w)
                return emb_init
            emb_init_sent = build_emb(pretrained_vectors, word2idx_sent).astype('float32')
            with open(os.path.join(data_dir, 'emb_sent.pkl'), 'wb') as f:
                pkl.dump(emb_init_sent, f)
            logger.info('Pretrained vectors has been created from {}'.format(pretrain_fn))

    # create target embedding from pretrained vectors
    target_emb_init_sent = np.zeros([len(target2idx_sent), dim_emb], dtype = float)
    if os.path.exists(os.path.join(data_dir, 'target_emb_sent.pkl')):
        logger.info('target embedding matrix already exisits in {}'.format(data_dir))
        target_emb_init_sent = pkl.load(open(os.path.join(data_dir, 'target_emb_sent.pkl'), 'rb'))
    else:
        for target in target2idx_sent:
            for word in target.split():
                #if word2idx_sent[word] in emb_init_sent:
                target_emb_init_sent[target2idx_sent[target]] += emb_init_sent[word2idx_sent[word]]
                #else:
                #    print(word2idx_sent[word])
            target_emb_init_sent[target2idx_sent[target]] /= max(1, len(target.split()))
        with open(os.path.join(data_dir, 'target_emb_sent.pkl'), 'wb') as f:
            pkl.dump(target_emb_init_sent, f)
            logger.info('target pretrained vectors has been created from {}'.format(pretrain_fn))
    return word2idx_sent, target2idx_sent, emb_init_sent, target_emb_init_sent

def load_data(data_dir):
    word2idx = pkl.load(open(os.path.join(data_dir, 'vocab.pkl')))
    embedding = pkl.load(open(os.path.join(data_dir, 'emb.pkl')))
    return word2idx, embedding



class MEMClassifier(BaseModel):
    def __init__(self, nwords, word2idx, target2idx, init_hid, init_std, init_lr, batch_size,  nhop, edim, mem_size, lindim, max_grad_norm, pad_idx, pre_trained_context_wt, pre_trained_target_wt):

        super(MEMClassifier, self).__init__()

        self.nwords = nwords
        self.init_hid = init_hid
        self.init_std = init_std
        self.batch_size = batch_size
        self.nhop = nhop
        self.edim = edim
        self.mem_size = mem_size
        self.lindim = lindim
        self.max_grad_norm = max_grad_norm
        self.pad_idx = pad_idx
        self.pre_trained_context_wt = pre_trained_context_wt
        self.pre_trained_target_wt = pre_trained_target_wt
        self.word2idx = word2idx
        self.target2idx = target2idx
        self.init_lr = init_lr


        self.hid = []

        self.loss = None
        self.step = None
        self.optim = None

        self.log_loss = []
        self.log_perp = []
    
    def __str__(self):
        return str(self.__dict__)

    def build_memory(self, mem_inputs):
      #self.global_step = tf.Variable(0, name="global_step")

      self.input = mem_inputs['input']
      self.context = mem_inputs['context']
      self.mask = mem_inputs['mask']

      self.A = tf.Variable(tf.random_uniform([self.nwords, self.edim], minval=-0.01, maxval=0.01))
      self.ASP = tf.Variable(tf.random_uniform([self.pre_trained_target_wt.shape[0], self.edim], minval=-0.01, maxval=0.01))
      self.C = tf.Variable(tf.random_uniform([self.edim, self.edim], minval=-0.01, maxval=0.01))
      self.C_B =tf.Variable(tf.random_uniform([1, self.edim], minval=-0.01, maxval=0.01))
      self.BL_W = tf.Variable(tf.random_uniform([2 * self.edim, 1], minval=-0.01, maxval=0.01))
      self.BL_B = tf.Variable(tf.random_uniform([1, 1], minval=-0.01, maxval=0.01))

      
      self.Ain_c = tf.nn.embedding_lookup(self.A, self.context)
      self.Ain = self.Ain_c

      self.ASPin = tf.nn.embedding_lookup(self.ASP, self.input)
      self.ASPout2dim = tf.reshape(self.ASPin, [-1, self.edim])
      self.hid.append(self.ASPout2dim)
      

      for h in range(self.nhop):
        '''
        Bi-linear scoring function for a context word and aspect term
        '''
        self.til_hid = tf.tile(self.hid[-1], [1, self.mem_size])
        self.til_hid3dim = tf.reshape(self.til_hid, [-1, self.mem_size, self.edim])
        self.a_til_concat = tf.concat(axis=2, values=[self.til_hid3dim, self.Ain])
        self.til_bl_wt = tf.tile(self.BL_W, [self.batch_size, 1])
        self.til_bl_3dim = tf.reshape(self.til_bl_wt, [self.batch_size,  2 * self.edim, -1])
        self.att = tf.matmul(self.a_til_concat, self.til_bl_3dim)
        self.til_bl_b = tf.tile(self.BL_B, [self.batch_size, self.mem_size])
        self.til_bl_3dim = tf.reshape(self.til_bl_b, [-1, self.mem_size, 1])
        self.g = tf.nn.tanh(tf.add(self.att, self.til_bl_3dim))
        self.g_2dim = tf.reshape(self.g, [-1, self.mem_size])
        self.masked_g_2dim = tf.add(self.g_2dim, self.mask)
        self.P = tf.nn.softmax(self.masked_g_2dim)
        self.probs3dim = tf.reshape(self.P, [-1, 1, self.mem_size])


        self.Aout = tf.matmul(self.probs3dim, self.Ain)
        self.Aout2dim = tf.reshape(self.Aout, [self.batch_size, self.edim])

        Cout = tf.matmul(self.hid[-1], self.C)
        til_C_B = tf.tile(self.C_B, [self.batch_size, 1])
        Cout_add = tf.add(Cout, til_C_B)
        self.Dout = tf.add(Cout_add, self.Aout2dim)

        if self.lindim == self.edim:
            self.hid.append(self.Dout)
        elif self.lindim == 0:
            self.hid.append(tf.nn.relu(self.Dout))
        else:
            F = tf.slice(self.Dout, [0, 0], [self.batch_size, self.lindim])
            G = tf.slice(self.Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
            K = tf.nn.relu(G)
            self.hid.append(tf.concat(axis=1, values=[F, K]))

    def create_placeholders(self, tag):
        with tf.name_scope('inputs'):
            plhs = dict()
            if tag == 'mem_inputs':
                plhs['input'] = tf.placeholder(tf.int32, [self.batch_size, 1], name="input")
                plhs['time'] = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
                plhs['target'] = tf.placeholder(tf.int64, [self.batch_size], name="target")
                plhs["context"] = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")
                plhs["mask"] = tf.placeholder(tf.float32, [self.batch_size, self.mem_size], name="mask")
                plhs["neg_inf"] = tf.fill([self.batch_size, self.mem_size], -1*np.inf, name="neg_inf")
            elif tag == 'y':
                plhs['y'] = tf.placeholder(tf.int64, [self.batch_size], name="y")
            elif tag == 'hyper':
                plhs['keep_rate'] = tf.placeholder(tf.float32, [], name='keep_rate')
            else:
                raise Exception('{} is not supported in create_placeholders'.format(tag))   
        return plhs

    def forward(self, mem_inputs):
        with tf.name_scope('forward'):
            self.build_memory(mem_inputs)
            self.W = tf.Variable(tf.random_uniform([self.edim, 3], minval=-0.01, maxval=0.01))
            self.z = tf.matmul(self.hid[-1], self.W)
            #logits = self.z
        return self.z

        
    def run(self, sess, train_data, test_data, n_iter, keep_rate, save_dir):
        
        #self.init_global_step()
        self.global_step = tf.Variable(0, name="global_step")
        input_placeholders = self.create_placeholders('mem_inputs')
        print("place_holders:" + str(input_placeholders))
        #y = self.create_placeholders['y']['y']
        self.y = tf.placeholder(tf.int64, [self.batch_size], name="y")
        
        logits = self.forward(input_placeholders)

        
        params = [self.A, self.C, self.C_B, self.W, self.BL_W, self.BL_B]
        with tf.name_scope('loss'):
            #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target))
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            self.loss = tf.reduce_sum(self.loss) 
        with tf.name_scope('train'):
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)
            self.lr = tf.Variable(self.init_lr)
            self.opt = tf.train.AdagradOptimizer(self.lr)
            grads_and_vars = self.opt.compute_gradients(self.loss,params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                    for gv in grads_and_vars]
        
            inc = self.global_step.assign_add(1)
            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(logits, axis=1), self.y)
            self.correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            self._acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        summary_loss = tf.summary.scalar('loss', self.loss)
        summary_acc = tf.summary.scalar('acc', self._acc)
        train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
        test_summary_op = tf.summary.merge([summary_loss, summary_acc])

        import time
        timestamp = str(int(time.time()))
        #_dir = save_dir + '/logs/' + str(timestamp) + '_' +  '_r' + str(self.init_lr) + '_b'  + '_l' + str(self.l2_reg)
        _dir = save_dir + '/logs/' + str(timestamp) + '_' +  '_r' + str(self.init_lr) + '_b'  + '_l'
        train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
        test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
        validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)
        print(self.pre_trained_target_wt[:10])
        sess.run(tf.global_variables_initializer())
        sess.run(self.A.assign(self.pre_trained_context_wt))
        sess.run(self.ASP.assign(self.pre_trained_target_wt))


        ##################
        # fetch and feed #
        ##################
        def fetch_results(data, test=False):
            source_data, source_loc_data, target_data, target_label = self.prepare_data(data)
            x = np.ndarray([self.batch_size, 1], dtype=np.int32)
            time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
            target = np.zeros([self.batch_size], dtype=np.int32) 
            context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
            mask = np.ndarray([self.batch_size, self.mem_size])
          
        
            context.fill(self.pad_idx)
            time.fill(self.mem_size)
            target.fill(0)
            mask.fill(-1.0*np.inf)


            for b in range(self.batch_size):
                 x[b][0] = target_data[b]
                 target[b] = target_label[b]
                 time[b,:len(source_loc_data[b])] = source_loc_data[b]
                 context[b,:len(source_data[b])] = source_data[b]
                 mask[b,:len(source_data[b])].fill(0)
                 #cur = cur + 1
            if not test:
                logits, _, loss, self.step, summary_op = sess.run([ self.z, self.optim,
                                                   self.loss,
                                                   self.global_step,
						   train_summary_op],
                                                   feed_dict={
                                                   self.input: x,
                                                   self.y: target,
                                                   self.context: context,
                                                   self.mask: mask})
                return logits, loss, self.step, summary_op
    
            if test:
                #raw_labels = []
                logits, loss, correct_pred, summary = sess.run([self.z, self.loss, self.correct_pred, test_summary_op], feed_dict={self.input: x,
                                                             self.y: target,
                                                             self.context: context,
                                                             self.mask: mask})
                #for b in range(self.batch_size):
                #    if raw_labels[b] == prediction[b]:
                #        acc += 1
                #print(np.sum(correct_pred))
                return logits, loss, correct_pred, summary

        max_acc = 0.
        for i in range(n_iter):
            train_acc, train_loss, train_cnt  = 0., 0., 0.
            for samples, in train_data:
               num = len(samples)
               logits, loss, step, summary = fetch_results(samples)
               train_summary_writer.add_summary(summary, step)
               _, _, _, gt = self.prepare_data(samples)
               train_acc += np.sum(np.argmax(logits,1) == gt)
               train_loss += loss * num
               train_cnt += num
            train_loss /= train_cnt
            train_acc /= train_cnt
            acc, loss, cnt = 0., 0., 0
            for samples, in test_data:
                num = len(samples)
                logits, _loss, correct_pred, summary = fetch_results(samples, test=True)
                #acc += np.sum(np.argmax(logits,1) == d)
                acc += correct_pred
                loss += _loss * num 
                cnt += num
            test_summary_writer.add_summary(summary, step)
            print('Iter {}: mini-batch train_loss={:.6f}, train acc={:.6f}, test_loss={:.6f}, test acc={:.6f}'.format(step, train_loss , train_acc, loss / cnt,  acc / cnt))
            if acc / cnt > max_acc:
                max_acc = acc / cnt
        print('Optimization Finished! Max acc={}'.format(max_acc))
    
    def prepare_data(self, samples): # , sentence_len, type_='', encoding='utf8'):
        sentence_len = self.mem_size
        sent_word2idx = self.word2idx
        target_word2idx = self.target2idx
        sentence_list = []
        location_list = []
        target_list = []
        polarity_list = []
        
        for sample in samples:
            # Get the segmented sentence list.
            words = []
            has_target = False
            for i in range(len(sample['tokens'])):
                if sample['tags'][i] == 'O':
                    words.append(sample['tokens'][i])
                elif not has_target:
                    words.append('$t$') 
                    has_target = True

            # Get the segmented target list. 
            target_word = [sample['tokens'][i] for i, _ in enumerate(sample['tokens']) if sample['tags'][i] != 'O']
            # Get the polarity symbol.
            y_dict = {'positive': 1, 'negative': 2, 'neutral': 0}
            polarity = y_dict[sample.get('polarity','positive')]

            sentence = " ".join(words).lower()
            target = " ".join(target_word).lower()
            
            sent_words = sentence.split()
            target_words = target.split()

            try:
                target_location = sent_words.index("$t$")
            except:
                print("sentence does not contain target element tag")
                exit()
            is_included_flag = 1
            id_tokenised_sentence = []
            location_tokenised_sentence = []
            for index, word in enumerate(sent_words):
                if word == "$t$":
                    continue
                try:
                    word_index = sent_word2idx[word]
                except:
                    print("id not found for word in the sentence")
                    exit()
                
                location_info = abs(index - target_location)
		#Mem_ABSA data load code
                #if word in self.embeddings:
                id_tokenised_sentence.append(word_index)
                location_tokenised_sentence.append(location_info)
	    #Mem_ABSA data load code
            #is_included_flag = 0
            #for word in target_words:
            #    if word in self.embeddings:
            #        is_included_flag = 1
            #    break
            try:
                target_index = target_word2idx[target]
            except:
                print(target)
                print("id not found for target")
                exit()

	    #Mem_ABSA data load code
            #if not is_included_flag:
            #    print(sentence)
            #    continue

            sentence_list.append(id_tokenised_sentence)
            location_list.append(location_tokenised_sentence)
            target_list.append(target_index)
            polarity_list.append(polarity)
        #print(sentence_list[:10])
        #print(location_list[:10])
        #print(target_list[:10])
        #print(polarity_list[:10])
        #print()
        return sentence_list, location_list, target_list, polarity_list

    
def main(_):
    from src.io.batch_iterator import BatchIterator
    train = pkl.load(open('../../../../data/se2014task06/tabsa-rest/train.pkl', 'rb'), encoding='latin')
    test = pkl.load(open('../../../../data/se2014task06/tabsa-rest/test.pkl', 'rb'), encoding='latin')
    
    fns = ['../../../../data/se2014task06/tabsa-rest/train.pkl',
            '../../../../data/se2014task06/tabsa-rest/dev.pkl',
            '../../../../data/se2014task06/tabsa-rest/test.pkl',]

    data_dir = '../0617'
    #data_dir = '/Users/wdxu//workspace/absa/TD-LSTM/data/restaurant/for_absa/'
    word2idx, target2idx, word_embedding, target_embedding = preprocess_data(fns, '../../../../data/glove.6B/glove.6B.300d.txt', data_dir)
    word_embedding = np.concatenate([word_embedding, np.zeros([1, FLAGS.embedding_dim])])
    #print(target_embedding[:10,:])
    train_it = BatchIterator(len(train), FLAGS.batch_size, [train], testing=False)
    test_it = BatchIterator(len(test), FLAGS.batch_size, [test], testing=False)
 
    #train_data = get_dataset(train, source_word2idx, target_word2idx, embeddings)
    #test_data = get_dataset(test, source_word2idx, target_word2idx, embeddings)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    with tf.Session(config=configproto) as sess:
        tf.global_variables_initializer().run()

        #model = MEMClassifier(word2idx=word2idx, 
        #        embedding_dim=FLAGS.embedding_dim, 
        #        n_hidden=FLAGS.n_hidden, 
        #        learning_rate=FLAGS.learning_rate, 
        #        n_class=FLAGS.n_class, 
        #        max_sentence_len=FLAGS.max_sentence_len, 
        #        l2_reg=FLAGS.l2_reg, 
        #        embedding=embedding,
        #        grad_clip=FLAGS.grad_clip)
        model = MEMClassifier(nwords=len(word2idx)+1,
                  word2idx = word2idx,
                  target2idx = target2idx,
                  init_hid=0.1,
                  init_std=0.01,
                  init_lr=0.01,
                  batch_size=FLAGS.batch_size,
                  nhop=9,
                  edim=FLAGS.embedding_dim,
                  mem_size=FLAGS.max_sentence_len,
                  lindim=300,
                  max_grad_norm=100,
                  pad_idx=len(word2idx),
                  pre_trained_context_wt=word_embedding,
                  pre_trained_target_wt=target_embedding)
        logger.info(model)            
        model.run(sess, train_it, test_it, FLAGS.n_iter, FLAGS.keep_rate, data_dir)
        logger.info(model)            
	

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 300, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('max_sentence_len', 79, 'max number of tokens per sentence')
    tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
    tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
    
    tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/train.raw', 'training file')
    tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/validate.raw', 'validating file')
    tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/test.raw', 'testing file')
    #tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_word_embedding_partial_100.txt', 'embedding file')
    #tf.app.flags.DEFINE_string('word_id_file_path', 'data/twitter/word_id.txt', 'word-id mapping file')
    tf.app.flags.DEFINE_string('type', 'TC', 'model type: ''(default), TD or TC')
    tf.app.flags.DEFINE_float('keep_rate', 0.5, 'keep rate')
    tf.app.flags.DEFINE_float('grad_clip', 5.0, 'gradient_clip')

    tf.app.run()
