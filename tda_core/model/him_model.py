#!/home/tops/bin/python
# -*- coding: utf-8 -*-
# vim:ts=4:sts=4:sw=4:et:fenc=utf8

import six
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
import json

from model_ops.dnn_fn import dnn_layer_with_input_fc, dnn_layer, dnn_logit_seq, DNN_ACTIVATION_FUNCTIONS, dnn_multihead_attention, keep_multihead_attention, dnn_multihead_attention_count, dnn_multihead_attention_count_without_forward
from utils.constant import *
from model_ops.metrics import variable_summaries, add_summary_gradient
from model_ops.attention import multihead_attention

def cosine(q,a):
    pooled_len_1 = tf.expand_dims(tf.sqrt(tf.reduce_sum(q * q, 1)), -1)
    pooled_len_2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(a * a, 1)), -1)
    pooled_mul_12 = tf.matmul(q, a, transpose_b=True)
    l = tf.matmul(pooled_len_1, pooled_len_2, transpose_b=True)
    score = tf.div(pooled_mul_12, l +1e-8, name="scores")
    return score

def pool_v1(ubb_combiner, value_i, name, axis=1):
    with variable_scope.variable_scope(name+'pool', partitioner=None):
        embedding_i_pool = []
        for method in ubb_combiner.split(','):
            if method == 'sum':
                ubbi = tf.reduce_sum(value_i, axis=axis)
            elif method == 'max':
                ubbi = tf.reduce_max(value_i, axis=axis)
            elif method == 'avg':
                ubbi = tf.divide(tf.reduce_sum(value_i, axis=1),
                                 tf.cast(tf.count_nonzero(value_i, axis=1), tf.float32) + 1e-8)
            embedding_i_pool.append(ubbi)
        embedding_i_pool_output = tf.concat(embedding_i_pool, -1)
    return embedding_i_pool_output

class HimModel():

    def __init__(self,FLAGS):
        self._FLAGS = FLAGS
        # To Be  refactor
        try:
            with open(FLAGS.transform_json, 'r') as fp:
                self._configDict = json.load(fp)
        except Exception as e:
            raise RuntimeError("open file fail :" + FLAGS.transform_json)

        self._dnn_activation = FLAGS.dnn_activation
        self._drop_out = FLAGS.drop_out
        self._dnn_l2_weight = FLAGS.dnn_l2_weight
        self._lr_op = FLAGS.lr_op
        self._dnn_lr = FLAGS.dnn_lr
        self._lr_decay_steps = FLAGS.lr_decay_steps
        self._lr_decay_rate = FLAGS.lr_decay_rate
        self._clip_gradients = FLAGS.clip_gradients
        self._ubb_combiner = FLAGS.ubb_combiner
        self._ubb_dim = FLAGS.ubb_dim

        self._logits_dimension = FLAGS.logits_dimension
        self._ema_decay = FLAGS.ema_decay

        # ========== parent scope name ==========
        self._dnn_item_parent_scope = 'dnn_item'
        self._dnn_user_parent_scope = 'dnn_user'

        self._batch_size = FLAGS.batch_size
        self._decision_layer_units = FLAGS.decision_layer_units
        self._ubb_time = ['3d','7d','14d','30d']
        self._rec_loss = 0

    def input_layer(self, featureColumnBuilder):
        with variable_scope.variable_scope('fea_col') as scope:
            self._feature_dict = featureColumnBuilder.getFutureColumnDict()
            self._dnn_user_feature_columns = featureColumnBuilder.getWideColumns()
            self._item_id_columns = featureColumnBuilder.getItemIdColumns()

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, is_training):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        # logits: [batch_size, n_class] unnormalized log-probs
        y = logits + tf.cond(is_training, lambda :self.sample_gumbel(tf.shape(logits)), lambda: tf.zeros_like(logits))
        return tf.nn.softmax(y / temperature)  # 每行之和为1

    def gumbel_softmax(self, logits, temperature, is_training, hard=False):
        """
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        # 返回值y.shape=(batchsize, n_class), 每行之和为1，每个数代表概率
        y = self.gumbel_softmax_sample(logits, temperature, is_training)
        if hard:
            # 将 y 转成one-hot向量，每一行最大值处为1，其余地方为0
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y  # y_hard = y
        return y

    def group_cluster(self, scope, user_emb_ori, user_emb_g, FLAGS, mode, activation, is_training, user_emb_num,
                      group_forward_layer_units=None, mask=None):
        with variable_scope.variable_scope(scope):
            with variable_scope.variable_scope("group_cluster"):
                group_embedding = tf.get_variable(name='group_emb_' + scope,
                                                  shape=[FLAGS.groups_num, FLAGS.group_embedding_num],
                                                  initializer=tf.contrib.layers.xavier_initializer()
                                                  , collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                                 ops.GraphKeys.MODEL_VARIABLES])
                with variable_scope.variable_scope("reconstruct"):
                    user_beita = tf.contrib.layers.fully_connected(inputs=user_emb_ori,
                                                                   num_outputs=FLAGS.groups_num,
                                                                   activation_fn=None,
                                                                   normalizer_fn=None,
                                                                   weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                   )
                    user_beita = tf.nn.softmax(user_beita)
                    user_mui = tf.matmul(user_beita, group_embedding)
                    user_resconstruct = tf.contrib.layers.fully_connected(inputs=user_mui,
                                                                          num_outputs=user_emb_num,
                                                                          activation_fn=tf.nn.tanh,
                                                                          normalizer_fn=None,
                                                                          weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                          )

                    with variable_scope.variable_scope("rec_loss"):
                        distance = cosine(user_resconstruct, user_emb_g)
                        loss_pos_resconstruct = tf.expand_dims(tf.diag_part(distance), -1)
                        loss_neg_resconstruct = tf.where(tf.abs(distance - loss_pos_resconstruct) < 0.0001,
                                                         distance - 1, distance)
                        loss_resconstruct = tf.maximum(0.0, 1 - loss_pos_resconstruct + loss_neg_resconstruct)
                        loss_rec = tf.reduce_sum(loss_resconstruct, -1, keep_dims=True) + 0.000001

                max_index = tf.argmax(user_beita, -1)
                max_value = tf.reduce_max(user_beita, -1, keep_dims=True)
                index_onehot = tf.cast(user_beita < max_value, tf.float32)
                group_user_emb = tf.matmul(index_onehot, group_embedding)

        return max_index, group_user_emb, loss_rec

    def net(self, mode, features, is_training):

        FLAGS = self._FLAGS
        activation = DNN_ACTIVATION_FUNCTIONS[FLAGS.dnn_activation]
        if FLAGS.drop_out is None:
            keep_prob = 1.0
        else:
            keep_prob = 1 - FLAGS.drop_out

        # item feature
        dnn_item_parent_scope = 'dnn_item'
        with variable_scope.variable_scope(dnn_item_parent_scope) as scope:
            item_id_emb = tf.contrib.layers.input_from_feature_columns(features, self._item_id_columns)
            item_emb = item_id_emb

        # user feature
        dnn_user_parent_scope = 'dnn_user'
        with variable_scope.variable_scope(dnn_user_parent_scope) as scope:
            user_emb = tf.contrib.layers.input_from_feature_columns(features, self._dnn_user_feature_columns)

        with variable_scope.variable_scope('pos'):
            clk_emb_list = []
            clk_orig_emb_list = []
            clk_pyramid_input = []
            clk_mask = []
            for time_i in self._ubb_time:
                with variable_scope.variable_scope(time_i):
                    ubb_embed_id_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict['user_clk_item_' + time_i + '_id'])
                    ubb_embed_value_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict['user_clk_item_' + time_i + '_value'])
                    value_i = tf.multiply(ubb_embed_id_i, ubb_embed_value_i)
                    value_i = tf.reshape(value_i, [tf.shape(item_emb)[0], -1, FLAGS.ubb_dim])

                    # ubb_embed_noclk_id_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict['user_noclk_item_' + time_i + '_id'])
                    # ubb_embed_noclk_value_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict['user_noclk_item_' + time_i + '_value'])
                    # value_noclk_i = pool_v1('sum,max', tf.reshape(tf.multiply(ubb_embed_noclk_id_i, ubb_embed_noclk_value_i), [tf.shape(item_emb)[0], -1, FLAGS.ubb_dim]), name='p0', axis=1)
                    # clk_emb_list.append(value_noclk_i)
                    # target = tf.expand_dims(value_noclk_i, 1)
                    # length = tf.count_nonzero(tf.reshape(ubb_embed_value_i, [tf.shape(item_emb)[0], -1, 1]), -1)
                    # value_i, w0 = dnn_multihead_attention(queries=target, keys=value_i,
                    #                                       key_length=length, scope='noclk-clk',
                    #                                       num_heads=1, keep_prob=keep_prob,
                    #                                       num_units=FLAGS.attention_num_units,
                    #                                       num_output_units=FLAGS.attention_num_output_units,
                    #                                       is_training=is_training,
                    #                                       num_units_forward=map(int,
                    #                                                             FLAGS.attention_num_units_forward.split(
                    #                                                                 ",")),
                    #                                       activation_fn=activation)
                    # value_i = tf.squeeze(value_i, axis=1)

                    clk_orig_emb_list.append(value_i)

                    with variable_scope.variable_scope('pool_fcn', partitioner=None):
                        embedding_i_pool_output = pool_v1('sum,max', value_i, name='p0', axis=1)
                        embedding_i_pool_output = dnn_layer(name='p_fc', net=embedding_i_pool_output, mode=mode,
                                                            hidden_units=FLAGS.attention_num_units_forward.split(","),
                                                            dropout=FLAGS.drop_out,
                                                            activation_fn=activation,
                                                            dnn_parent_scope='p_fc',
                                                            is_training=is_training,
                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                scale=FLAGS.dnn_l2_weight),
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                            FLAGS=FLAGS)
                        embedding_i_pool_output = tf.contrib.layers.batch_norm(embedding_i_pool_output, is_training=is_training)

                    with variable_scope.variable_scope('trans', partitioner=None):
                        ubb_embed_value_i_reshape = tf.reshape(ubb_embed_value_i, [tf.shape(user_emb)[0], -1, 1])

                        embedding_i_concat_counts = tf.slice(ubb_embed_value_i_reshape, [0, 0, 0], [-1, FLAGS.ubb_pos_slice_len, -1])
                        ubb_seq_length = tf.count_nonzero(tf.squeeze(embedding_i_concat_counts, axis=-1), axis=-1)
                        value_i = tf.slice(value_i, [0, 0, 0], [-1, FLAGS.ubb_pos_slice_len, -1])

                        outputs, w0 = dnn_multihead_attention(queries=value_i, keys=value_i,
                                                              key_length=ubb_seq_length, scope='seq_t',
                                                              num_heads=1, keep_prob=keep_prob,
                                                              num_units=FLAGS.attention_num_units,
                                                              num_output_units=FLAGS.attention_num_output_units,
                                                              is_training=is_training,
                                                              num_units_forward=map(int,
                                                                                    FLAGS.attention_num_units_forward.split(
                                                                                        ",")),
                                                              activation_fn=activation)
                        embedding_gru = pool_v1("avg,max,sum", outputs, "p", 1)

                        l = tf.reduce_sum(tf.squeeze(embedding_i_concat_counts, axis=-1), -1, keep_dims=True)
                        one = tf.ones_like(l)
                        l = tf.where(l > 0, one, l)
                        clk_mask.append(l)

                    embedding_pool_att = tf.concat([embedding_i_pool_output, embedding_gru], -1)
                    clk_emb_list.append(embedding_pool_att)
                    clk_pyramid_input.append(embedding_gru)

            with variable_scope.variable_scope('pyramid', partitioner=None):
                tensors = []
                for tensori in clk_emb_list:
                    tensors.append(tf.expand_dims(tensori, axis=1))
                input_to_gru = tf.concat(tensors, axis=1)

                leng = len(self._ubb_time) * tf.ones_like(ubb_seq_length)
                outputs, w0 = dnn_multihead_attention(queries=input_to_gru, keys=input_to_gru,
                                                      key_length=leng, scope='seq_t',
                                                      num_heads=1, keep_prob=keep_prob,
                                                      num_units=FLAGS.attention_num_units,
                                                      num_output_units=FLAGS.attention_num_output_units,
                                                      is_training=is_training,
                                                      num_units_forward=map(int, FLAGS.attention_num_units_forward.split(",")),
                                                      activation_fn=activation)

                clk_pyramid_output = tf.split(outputs, len(self._ubb_time), axis=1)
                clk_emb_list.append(tf.reshape(outputs, [tf.shape(outputs)[0], len(clk_emb_list) * FLAGS.attention_num_units]))

            with variable_scope.variable_scope('orig', partitioner=None):
                m0 = clk_orig_emb_list[0]
                for j in range(1, len(self._ubb_time)):
                    with variable_scope.variable_scope(str(j), partitioner=None):
                        m1 = clk_orig_emb_list[j]
                        m = tf.concat([m0, m1], axis=1)
                        p_emb = pool_v1('sum,max', m, name='ubb_orig')
                        clk_emb_list.append(p_emb)
                        m0 = m
            pos_emb = tf.concat(clk_emb_list, -1)
            with variable_scope.variable_scope("mlp"):
                pos_emb_for_att = tf.contrib.layers.fully_connected(inputs=pos_emb,
                                                                    num_outputs=256,
                                                                    activation_fn=activation,
                                                                    normalizer_fn=None,
                                                                    weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                    )

        with variable_scope.variable_scope("group"):
            group_user_emb_list = []
            group_user_emb_list_clk = []
            for i in range(len(self._ubb_time)):
                user_emb_ori = clk_pyramid_input[i]  # px
                user_emb_g = tf.squeeze(clk_pyramid_output[i], axis=1)  # pz
                clk_mask_i = clk_mask[i]
                max_index, group_user_emb, loss_rec = self.group_cluster(scope='g_' + str(i),
                                                                         user_emb_ori=user_emb_ori,
                                                                         user_emb_g=user_emb_g, FLAGS=FLAGS,
                                                                         mode=mode, activation=activation,
                                                                         is_training=is_training,
                                                                         user_emb_num=int(
                                                                             FLAGS.attention_num_units_forward.split(
                                                                                 ",")[-1])
                                                                         , mask=clk_mask_i)
                self._rec_loss += loss_rec

                group_user_emb_list.append(group_user_emb)
                v1 = tf.expand_dims(group_user_emb, axis=1)
                group_user_emb_list_clk.append(v1)
            semi_emb = tf.concat(group_user_emb_list, -1)
            with variable_scope.variable_scope("mlp"):
                semi_emb_for_att = tf.contrib.layers.fully_connected(inputs=semi_emb,
                                                                     num_outputs=256,
                                                                     activation_fn=activation,
                                                                     normalizer_fn=None,
                                                                     weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                     )

        with variable_scope.variable_scope("ta"):
            target = tf.expand_dims(item_emb, 1)
            outputs = tf.stack([semi_emb_for_att, pos_emb_for_att], 1)
            all_length = 2 * tf.ones([tf.shape(item_id_emb)[0], ])
            ubp_ubc, w0 = dnn_multihead_attention(queries=target, keys=outputs,
                                                  key_length=all_length, scope='ta',
                                                  num_heads=1, keep_prob=keep_prob,
                                                  num_units=FLAGS.attention_num_units,
                                                  num_output_units=FLAGS.attention_num_output_units,
                                                  is_training=is_training,
                                                  num_units_forward=map(int,
                                                                        FLAGS.attention_num_units_forward.split(
                                                                            ",")),
                                                  activation_fn=activation)
            ubp_ubc = tf.squeeze(ubp_ubc, axis=1)

        output_emb = tf.concat([item_emb, user_emb, pos_emb, semi_emb, ubp_ubc], axis=-1)  # [(batch_size, 2)
        with variable_scope.variable_scope('forward') as scope:
            output_emb = dnn_layer(name='forward', net=output_emb, mode=mode,
                                   hidden_units=FLAGS.decision_layer_units.split(","),
                                   dropout=FLAGS.drop_out,
                                   activation_fn=activation,
                                   dnn_parent_scope=dnn_user_parent_scope,
                                   is_training=is_training,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.dnn_l2_weight),
                                   kernel_initializer=tf.contrib.layers.xavier_initializer()
                                   , FLAGS=FLAGS)
            output_emb = tf.contrib.layers.batch_norm(output_emb, is_training=is_training)

        with variable_scope.variable_scope("logits") as scope:
            deep_logits = tf.contrib.layers.fully_connected(inputs=output_emb,
                                                            num_outputs=1,
                                                            activation_fn=None,
                                                            normalizer_fn=None,
                                                            weights_initializer=tf.contrib.layers.xavier_initializer()
                                                            )
            self._predict_score = tf.sigmoid(deep_logits)

        self._rank_predict = tf.identity(self._predict_score, name="rank_predict")
        self._logits = deep_logits


    def opt(self, loss, global_step):

        dnn_learning_rate = self._dnn_lr
        dnn_optimizer = tf.train.AdamOptimizer(learning_rate=dnn_learning_rate)
        train_op = dnn_optimizer.minimize(loss, global_step=global_step)
        self._train_op = train_op
        self._dnn_learning_rate = dnn_learning_rate

    def loss(self, features, labels, logits):
        raw_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels['click'], tf.float32),
                                                           logits=logits,
                                                           name='logits_cross_entropy')

        weighted_loss = raw_loss
        ltr_loss = weighted_loss + self._FLAGS.rate_for_recloss*self._rec_loss

        reduce_loss = tf.reduce_mean(ltr_loss)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = reduce_loss + reg_loss
        self._loss = loss

        # use moving average loss to indicate training loss
        ema = tf.train.ExponentialMovingAverage(decay=self._ema_decay, zero_debias=True)
        self._loss_ema_op = ema.apply([reduce_loss, reg_loss, loss])
        self._avg_loss = ema.average(loss)

    def metric(self, labels, predict_score):
        _, self._ctr_auc_op = tf.metrics.auc(labels=tf.cast(labels['click'], tf.bool), predictions=predict_score)

    def build(self, mode, features, labels, feature_column_builder, global_step=None, is_training=True):

        self._mode = mode
        self._feature_column_builder = feature_column_builder
        self._global_step = global_step
        FLAGS = self._FLAGS
        self._feature_column_builder.buildColumns()

        self.input_layer(self._feature_column_builder)
        self.net(mode, features, is_training)
        self.loss(features, labels, self._logits)
        self.metric(labels, self._predict_score)
        if mode == ModeKeys.EVAL:
            return self._logits, self._predict_score, self._ctr_auc_op, self._logits
        self.opt(self._loss, global_step)
        if self._mode == ModeKeys.TRAIN:
            return self._loss, self._train_op, self._ctr_auc_op, self._loss_ema_op, self._avg_loss, self._logits