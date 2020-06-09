
import json

import tensorflow as tf
import traceback
import datetime
from utils.constant import *
import importlib
from model.hhin_model import HhinModel
from model_ops.FeatureColumnBuilder_v3 import FeatureColumnBuilder

def dense2sparse(FLAGS,arr_tensor, str_0, len):
    # arr_tensor = tf.constant(np.array(arr))
    arr_idx = tf.where(tf.not_equal(arr_tensor, str_0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), [FLAGS.batch_size, len])
    return arr_sparse

def StringtoKV(features, conf):
    for fc in conf['feature_columns']:
        if fc.get(TRANSFORM_NAME) == "string_value":
            flag = tf.constant([[chr(5).join(['item' + chr(6) + '0' for i in range(int(fc.get(MAX_LENGTH)))])]])
            feature_flag = tf.concat([flag, tf.reshape(features[fc.get(INPUT_NAME)], [-1, 1])], 0)
            feature = tf.sparse_to_dense(sparse_indices=tf.string_split(tf.reshape(feature_flag, [-1, ]), chr(5)).indices,
                                         sparse_values=tf.string_split(tf.reshape(feature_flag, [-1, ]), chr(5)).values
                                         , output_shape=tf.string_split(tf.reshape(feature_flag, [-1, ]), chr(5)).dense_shape
                                         , default_value='item' + chr(6) + '0')
            # feature = feature[:,:int(fc.get(MAX_LENGTH))]
            feature1, feature2 = tf.split(tf.reshape(tf.string_split(tf.reshape(feature, [-1, ]), chr(6)).values, [-1, 2]), 2, -1)
            feature1 = tf.reshape(feature1, [tf.shape(feature_flag)[0], -1])[1:, ]
            feature2 = tf.reshape(feature2, [tf.shape(feature_flag)[0], -1])[1:, ]
            features[fc.get(INPUT_NAME) + '_id'] = tf.reshape(feature1, [-1, 1])    #batch_size*max_length
            features[fc.get(INPUT_NAME) + '_value'] = tf.reshape(tf.string_to_number(feature2, out_type=tf.float32),
                                                                 [-1, 1])
    return features


def get_model_instance(FLAGS):

    model_name = FLAGS.model_name
    pathstr = ''
    model_name_exclude_path = model_name
    if '.' in model_name:
        pathstr = '.'.join(model_name.split('.')[:-1]) + '.'
        model_name_exclude_path = model_name.split('.')[-1]
    pos = 0
    for ch in model_name_exclude_path:
        if ch.isupper():
            if pos != 0 :
                pathstr += '_'
        pos += 1
        pathstr += ch.lower()

    print "get_model_instance", pathstr, model_name_exclude_path
    model_lib = importlib.import_module("model."+pathstr)
    model_cls = getattr(model_lib,model_name_exclude_path)
    model_instance = model_cls(FLAGS)
    return model_instance

class DistributedScheduler():

    def train(self, FLAGS):
        filename_queue = tf.train.string_input_producer(["../data/data_musical_instruments_info_train.csv"], num_epochs=FLAGS.num_epoch)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        with open(FLAGS.transform_json, 'r') as fp:
            self._configDict = json.load(fp)

        record_defaults = [['']]*12
        unique,labels,userid,user_item_clk_1m,user_item_clk_6m,user_item_clk_1y,user_item_clk_3y,\
        user_activation_level,item_id,brand_id,price_level,leaf_cat = tf.decode_csv(value, record_defaults)

        unique, labels, userid, user_item_clk_1m, user_item_clk_6m, user_item_clk_1y, user_item_clk_3y, \
        user_activation_level, item_id, brand_id, price_level, leaf_cat = tf.train.shuffle_batch([unique,labels,userid,user_item_clk_1m,user_item_clk_6m,user_item_clk_1y,user_item_clk_3y,\
        user_activation_level,item_id,brand_id,price_level,leaf_cat], batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=4000, num_threads=2)

        features = {'user_item_clk_1m': user_item_clk_1m, 'user_item_clk_6m': user_item_clk_6m,
                    'user_item_clk_1y': user_item_clk_1y,
                    'user_item_clk_3y': user_item_clk_3y, 'user_activation_level': user_activation_level,
                    'item_id': item_id,
                    'brand_id': brand_id, 'price_level': price_level, 'leaf_cat': leaf_cat}

        model_instance = HhinModel(FLAGS)

        features = StringtoKV(features, self._configDict)
        self._features = features

        labels = tf.string_split(labels, ';')
        labels = tf.reshape(tf.sparse_tensor_to_dense(labels, default_value='0'), [tf.shape(labels)[0], 4])
        clk = tf.reshape(tf.string_to_number(labels[:, 0], tf.int64), [tf.shape(labels)[0], 1])
        self._labels = {'click':clk}
        self._feature_column_builder = FeatureColumnBuilder(self._configDict, FLAGS, features)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        loss_op, optimizer, ctr_auc_op, loss_ema_op, avg_loss, logits = \
            model_instance.build( ModeKeys.TRAIN, features, self._labels, self._feature_column_builder, global_step, is_training=True)

        summaryOp = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=20))
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_checkpoint_secs=FLAGS.checkpoint_sec,
                                               save_summaries_steps=FLAGS.summary_step,
                                               scaffold=scaffold) as mon_sess:
            mon_sess.run(init_op)
            mon_sess.run(local_init_op)
            step = 0
            try:  # maybe the last data slice is less than a batch, reshape function then raises exception
                print "start train"
                while not mon_sess.should_stop():
                    step += 1

                    if step % FLAGS.summary_step == 0:
                        _, loss, g_step, _, avg_loss_val, ctr_auc, summary = mon_sess.run(
                            [optimizer, loss_op, global_step, loss_ema_op, avg_loss, ctr_auc_op, summaryOp])
                    else:
                        _, loss, g_step, _, avg_loss_val, ctr_auc = mon_sess.run(
                            [optimizer, loss_op, global_step, loss_ema_op, avg_loss, ctr_auc_op])
            except:
                # summaryWriter.close()
                traceback.print_exc()
                print("train get exception at step=%d" % step)

    def eval(self, FLAGS):
        filename_queue = tf.train.string_input_producer(["../data/data_musical_instruments_info_test.csv"],
                                                        num_epochs=1)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        with open(FLAGS.transform_json, 'r') as fp:
            self._configDict = json.load(fp)

        record_defaults = [['']] * 12
        unique, labels, userid, user_item_clk_1m, user_item_clk_6m, user_item_clk_1y, user_item_clk_3y, \
        user_activation_level, item_id, brand_id, price_level, leaf_cat = tf.decode_csv(value, record_defaults)

        unique, labels, userid, user_item_clk_1m, user_item_clk_6m, user_item_clk_1y, user_item_clk_3y, \
        user_activation_level, item_id, brand_id, price_level, leaf_cat = tf.train.shuffle_batch(
            [unique, labels, userid, user_item_clk_1m, user_item_clk_6m, user_item_clk_1y, user_item_clk_3y, \
             user_activation_level, item_id, brand_id, price_level, leaf_cat], batch_size=FLAGS.batch_size,
            capacity=20000, min_after_dequeue=4000, num_threads=2)

        features = {'user_item_clk_1m': user_item_clk_1m, 'user_item_clk_6m': user_item_clk_6m,
                    'user_item_clk_1y': user_item_clk_1y,
                    'user_item_clk_3y': user_item_clk_3y, 'user_activation_level': user_activation_level,
                    'item_id': item_id,
                    'brand_id': brand_id, 'price_level': price_level, 'leaf_cat': leaf_cat}

        model_instance = HhinModel(FLAGS)

        features = StringtoKV(features, self._configDict)
        self._features = features

        labels = tf.string_split(labels, ';')
        labels = tf.reshape(tf.sparse_tensor_to_dense(labels, default_value='0'), [tf.shape(labels)[0], 4])
        clk = tf.reshape(tf.string_to_number(labels[:, 0], tf.int64), [tf.shape(labels)[0], 1])
        self._labels = {'click': clk}
        self._feature_column_builder = FeatureColumnBuilder(self._configDict, FLAGS, features)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        logits, predict_score, ctr_auc_op, deep_logits = \
            model_instance.build(ModeKeys.EVAL, features, self._labels, self._feature_column_builder, global_step,
                                 is_training=False)

        if FLAGS.model_num and FLAGS.model_num!='':
            if FLAGS.model_num.isdigit():
                pre_train_saver = tf.train.Saver()
                scaffold = tf.train.Scaffold(
                    init_fn=lambda scaffold, sess: pre_train_saver.restore(sess, FLAGS.checkpoint_dir+'model.ckpt-'+str(FLAGS.model_num)))
                checkpoint_dir = None
                print('predict checkpoint_dir: %s, start to load model at %s' % ( FLAGS.checkpoint_dir+'model.ckpt-'+str(FLAGS.model_num), datetime.datetime.now()))
            else:
                pre_train_saver = tf.train.Saver()
                scaffold = tf.train.Scaffold(
                    init_fn=lambda scaffold, sess: pre_train_saver.restore(sess, str(FLAGS.model_num)))
                checkpoint_dir = None
        else:
            scaffold = None
            checkpoint_dir = FLAGS.checkpoint_dir
            print('predict checkpoint_dir: %s, start to load model at %s' % ( FLAGS.checkpoint_dir, datetime.datetime.now()))
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None,
                                               scaffold=scaffold) as mon_sess:
            step = 0
            try:
                while not mon_sess.should_stop():
                    step += 1
                    ctr_auc, _, logits_v, predict_score_val = mon_sess.run([ctr_auc_op, logits, predict_score])
                    if step % FLAGS.print_step == 0:
                        print('predict: step %d done, ctr auc %s' % (step, ctr_auc),
                              datetime.datetime.now())
            except:
                traceback.print_exc()
            print("train get exception at step=%d" % step)
