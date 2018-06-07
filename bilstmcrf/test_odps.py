# -*- coding: utf-8 -*
import os
os.system('pip install jieba')

import tensorflow as tf
import numpy as np
#import data_helper
import math
import re
import config
import env_processor
import odps_writer_partition
import jieba
import sys
import itertools
import multiprocessing



#record_key_list = ['id', 'readable_record_text', 'readable_record_text_with_space', 'server_text', 'customer_text',
#                   'creator', 'gmt_create', 'modifier', 'gmt_modified', 'topic', 'subs_id', 'lead_no',
#                   'channel_code', 'dealer', 'call_time', 'start_time', 'release_time', 'record_file_osspath',
#                   'content', 'call_id', 'get_record_flag', 'record_text', 'partner_key', 'task_id',
#                   'retry_number', 'ch_cus_id']

record_key_list =['rowkey', 'std_category', 'shop_name', 'brand_name', '2nd_category', 'city_name', 'country_name',
            'comment_txt','description']

'''
flags =tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.1,'the learning rate')
flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
flags.DEFINE_integer('vocabulary_size',23429,'vocabulary_size')
flags.DEFINE_integer('embedding_dim',64,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',64,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
#flags.DEFINE_string('dataset_path','data/subj0.pkl','dataset path')
flags.DEFINE_integer('max_len',100,'max_len of training sentence')
flags.DEFINE_integer('valid_num',100,'epoch num of validation')
flags.DEFINE_integer('checkpoint_num',1000,'epoch num of checkpoint')
flags.DEFINE_float('init_scale',1/math.sqrt(64 / 2),'init scale')
flags.DEFINE_integer('class_num',12,'class num')
flags.DEFINE_float('keep_prob',0.5,'dropout rate')
flags.DEFINE_integer('num_epoch',20,'num epoch')
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')
flags.DEFINE_integer('check_point_every',1,'checkpoint every num epoch ')

class Config(object):
    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=FLAGS.vocabulary_size
    embed_dim=FLAGS.embedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    class_num=FLAGS.class_num
    keep_prob=FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size=20000
    num_step = FLAGS.max_len
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num=FLAGS.valid_num
    out_dir=FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every

def test_step(sentence, batch_size):
    tf.reset_default_graph()
    eval_config=Config()
    eval_config.keep_prob=1.0
    eval_config.batch_size = batch_size
    test_data = data_helper.load_test_data(sentence, FLAGS.max_len, eval_config.batch_size)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.variable_scope("model"):
            model = RNN_Model(config=eval_config, is_training=False, pretrained_embedding=None)

        #model_file = tf.train.latest_checkpoint("./runs/NewCheckpoint")
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, "./runs/NewCheckpoint/model-7580")
        x,y,mask_x= test_data
        feed_dict={}
        feed_dict[model.input_data]=x
        feed_dict[model.mask_x]=mask_x
        model.assign_new_batch_size(sess,len(x))
        fetches = [model.prob]
        state = sess.run(model._initial_state)

        #print state
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        prediction = sess.run(fetches,feed_dict)
    sess.close()
    return prediction[0].tolist()
        #for pred in prediction[0][:,1].tolist():
        #    print pred
               #line.split("\t")[-2]

'''

class Config():
    batch_size=20000

def main(block_id, block_num):
    prediction_name = [
    "便利店超市",
    "珠宝钟表饰品",
    "百货商场",
    "服装箱包饰品",
    "礼品纪念品伴手礼",
    "家居家具",
    "药店、保健品、美妆",
    "文化艺术店",
    "运动户外用品",
    "数码电器商店",
    "儿童",
    "枪支器械"
    ]

    odps_data_params = config.odps_data_params.copy()
    #odps_data_params['partition']='dt=20180506'
    data_count = env_processor.make_env(odps_data_params).count()
    per_thread_size = data_count / block_num
    range_start = per_thread_size * block_id
    range_end = min(per_thread_size * (block_id + 1), data_count)
    odps_data_params.update({"range_start": range_start, "range_end":range_end})
    sample_iter = env_processor.make_env(odps_data_params)

    odps_results_params = config.odps_results_params.copy()
    #odps_results_params['partition']='dt=20180506'

    test_config = Config()
    #result_writer = odps_writer_partition.make_writer(odps_results_params)
    data_size = range_end - range_start
    #VOCAB_SIZE = word2vector_helpers.load_vocab_size(FLAGS.embedding_dim)
    #pretrained_word2vec_matrix = word2vector_helpers.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)
    i = 0
    pos = 0
    while pos < data_size:
        print pos, data_size
        bs = min(test_config.batch_size, data_size - pos)
        sample = list(itertools.islice(sample_iter, pos, pos + bs))
        print(bs)
        print(type(sample))
        print(sample[0])
        print(sample[0][0]['comment_txt'])
        return
        # predict process
        if len(sample) == 0:
            print "sample's length is zero"
            i += 1
            pos += test_config.batch_size
            continue
        records, feature_list =[s[0] for s in sample] , [s[1] for s in sample]
        #print feature_list
#            print feature_list
        #text preprocess
        #prob = test_step(feature_list, bs)
        #prob  = [[0,0,0] for b in range(bs)]
        pred_list = []
        prob_list = []
        for pred in prob:
            prob_list.append(np.max(pred))
            pred_list.append(np.argmax(pred))
        # write a record
        rec_written = []
        for x, record in enumerate(records):
            rec_tmp = []
            if prob_list[x] < 0.25:
                rec_tmp.append("UNSEEN")
            else:
                rec_tmp.append(prediction_name[int(pred_list[x])])
            for k in record_key_list:
                rec_tmp.append(record[k])
            rec_written.append(rec_tmp)
        #result_writer.write_record(rec_written, block_id)

        # log info
        print("Write %dth batch record in block %d successfully." % (i + 1, block_id))
        i += 1
        pos += test_config.batch_size
    print("Write table successfully")



if __name__ == "__main__":
    print('fsfdsfs')
    #cpu_count = multiprocessing.cpu_count()
    cpu_count = 2
    predict_process = []
    for i in range(cpu_count - 1):
        p = multiprocessing.Process(target=lambda: main(i, cpu_count - 1))
        p.start()
        predict_process.append(p)

    for p in predict_process:
        p.join()



  #  with out_odps_table.open_writer(partition=params['out_partition'], blocks=range(thread_num)) as out_odps_writer:
              #  coord = tf.train.Coordinator()
                      #  running_threads = []
                              #  per_thread_size = params['total_sample'] / thread_num
                                      #  for evaluator in evaluators:
                                                      #  evaluator_fn = lambda: evaluator.run(out_odps_writer, sess)
                                                                  #  t = threading.Thread(target=evaluator_fn)
                                                                              #  t.start()
                                                                                          #  running_threads.append(t)
                                                                                                      #  time.sleep(1)
                                                                                                              #  # 等待所有线程结束
                                                                                                                      #  coord.join(running_threads)



