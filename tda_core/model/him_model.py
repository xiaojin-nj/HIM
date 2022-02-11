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

def pool(FLAGS, value_i, name, axis=1):
    with variable_scope.variable_scope(name+'pool', partitioner=None):
        embedding_i_pool = []
        for method in FLAGS.ubb_combiner.split(','):
            if method == 'sum':
                ubbi = tf.reduce_sum(value_i, axis=axis)
            elif method == 'max':
                ubbi = tf.reduce_max(value_i, axis=axis)
            elif method == 'avg':
                ubbi = tf.divide(tf.reduce_sum(value_i, axis=1),
                                 tf.cast(tf.count_nonzero(value_i, axis=1), tf.float32))
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
        self._ubb_time = ['1m', '6m', '1y', '3y']
        self._ubb_jfy_clk_feature = ["user_item_clk_1m", "user_item_clk_6m", "user_item_clk_1y", "user_item_clk_3y"]

    def input_layer(self, featureColumnBuilder):
        with variable_scope.variable_scope('fea_col') as scope:
            self._feature_dict = featureColumnBuilder.getFutureColumnDict()

            self._dnn_user_feature_columns = featureColumnBuilder.getWideColumns()
            self._ubb_embedding_id = featureColumnBuilder.getUbbEmbeddingID()
            self._ubb_embedding_value = featureColumnBuilder.getUbbEmbeddingValue()

            self._feature_colum_dict = featureColumnBuilder.getFutureColumnDict()
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

    def group_cluster(self, scope, user_emb_ori, user_emb_g, FLAGS, mode, activation, is_training, user_emb_num, group_forward_layer_units=None):
        if group_forward_layer_units is None:
            group_forward_layer_units = str(FLAGS.group_embedding_num) + ',' + str(FLAGS.group_embedding_num)

        with variable_scope.variable_scope(scope):
            with variable_scope.variable_scope("g_clu"):
                group_embedding = tf.get_variable(name='group_emb_'+scope,
                                                    shape=[FLAGS.groups_num, FLAGS.group_embedding_num],
                                                    initializer=tf.contrib.layers.xavier_initializer()
                                                    , collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                                   ops.GraphKeys.MODEL_VARIABLES])
                with variable_scope.variable_scope("rec"):
                    user_beita = tf.contrib.layers.fully_connected(inputs=user_emb_ori,
                                                                   num_outputs=FLAGS.groups_num,
                                                                   activation_fn=None,
                                                                   normalizer_fn=None,
                                                                   weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                   )
                    user_beita = self.gumbel_softmax(user_beita, FLAGS.temperature, is_training)
                    user_mui = tf.matmul(user_beita, group_embedding)
                    user_resconstruct = tf.contrib.layers.fully_connected(inputs=user_mui,
                                                                          num_outputs=user_emb_num,
                                                                          activation_fn=tf.nn.tanh,
                                                                          normalizer_fn=None,
                                                                          weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                          )

                    with variable_scope.variable_scope("recloss"):
                        distance = cosine(user_resconstruct, user_emb_g)
                        # distance = tf.matmul(user_resconstruct, user_emb_ori, transpose_b=True)
                        loss_pos_resconstruct = tf.expand_dims(tf.diag_part(distance), -1)
                        loss_neg_resconstruct = tf.where((distance - loss_pos_resconstruct)<0.0001, distance - 1,
                                                         distance)
                        loss_resconstruct = tf.maximum(0.0, 1 - loss_pos_resconstruct + loss_neg_resconstruct)
                        loss_rec = tf.reduce_sum(loss_resconstruct, -1, keep_dims=True)

            # # user_emb_g_stoped = tf.stop_gradient(user_emb_g)
            # with variable_scope.variable_scope("g_pre"):
            #     z_u = dnn_layer(name='forw', net=user_emb_ori, mode=mode,
            #                     hidden_units=group_forward_layer_units.split(","),
            #                     dropout=FLAGS.drop_out,
            #                     activation_fn=activation,
            #                     dnn_parent_scope='forw',
            #                     is_training=is_training,
            #                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
            #                         scale=FLAGS.dnn_l2_weight),
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer()
            #                     , FLAGS=FLAGS)

                max_index = tf.argmax(user_beita, -1)
                # max_index_onehot = tf.one_hot(max_index, depth=FLAGS.groups_num)
                # logits_cluster = tf.matmul(z_u, group_embedding, transpose_b=True)
                #
                # uu_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_cluster, labels=max_index_onehot)
                group_user_emb = tf.nn.embedding_lookup(group_embedding, max_index)

        return max_index, group_user_emb, loss_rec

    def net(self, mode, features, is_training):

        FLAGS = self._FLAGS
        activation = DNN_ACTIVATION_FUNCTIONS[FLAGS.dnn_activation]
        if FLAGS.drop_out is None:
            keep_prob = 1.0
        else:
            keep_prob = 1 - FLAGS.drop_out

        # Build DNN Logits.
        dnn_item_parent_scope = 'dnn_item'
        with variable_scope.variable_scope(dnn_item_parent_scope, values=tuple(six.itervalues(features))) as scope:
            activation = DNN_ACTIVATION_FUNCTIONS[FLAGS.dnn_activation]
            with variable_scope.variable_scope(
                    dnn_item_parent_scope + 'input',
                    values=tuple(six.itervalues(features))):
                item_id_emb = tf.contrib.layers.input_from_feature_columns(
                    features, self._item_id_columns)
                item_emb = item_id_emb
                input_i = item_emb

        item_emb_to_be = item_id_emb

        dnn_user_parent_scope = 'dnn_user'

        if not self._dnn_user_feature_columns:
            user_emb = None
            input_u = None
            user_profile_emb = None
        else:
            with variable_scope.variable_scope(dnn_user_parent_scope, values=tuple(six.itervalues(features))) as scope:

                with variable_scope.variable_scope(
                        dnn_user_parent_scope + 'input',
                        values=tuple(six.itervalues(features))):
                    net = tf.contrib.layers.input_from_feature_columns(
                        features, self._dnn_user_feature_columns)
                    user_emb = net
                    input_u = user_emb
                    user_profile_emb = user_emb

        ubb_scope_name = 'user_item'
        clk_jfy_scope = 'clk'
        with variable_scope.variable_scope('ubb_pos', values=tuple(six.itervalues(features))):

            with variable_scope.variable_scope(clk_jfy_scope, values=tuple(six.itervalues(features))):
                clk_jfy_emb = []
                clk_jfy_orig_emb = []
                clk_jfy_values = []
                clk_jfy_pyramid = []
                for time_i in self._ubb_time:
                    with variable_scope.variable_scope(str(time_i), values=tuple(six.itervalues(features))):
                        with variable_scope.variable_scope('input', values=tuple(six.itervalues(features))):
                            ubb_embed_id_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict[ubb_scope_name+'_'+clk_jfy_scope+'_'+str(time_i)+'_id'])
                            ubb_embed_value_i = tf.contrib.layers.input_from_feature_columns(features, self._feature_dict[ubb_scope_name+'_'+clk_jfy_scope+'_'+str(time_i)+'_value'])
                            value_i = tf.multiply(ubb_embed_id_i, ubb_embed_value_i)
                            value_i = tf.reshape(value_i, [tf.shape(user_emb)[0], -1, FLAGS.ubb_dim])
                            clk_jfy_orig_emb.append(value_i)

                        with variable_scope.variable_scope('pool_fcn', partitioner=None):
                            embedding_i_pool_output = pool(FLAGS, value_i, name='p0', axis=1)
                            embedding_i_pool_output = dnn_layer(name='p_fc', net=embedding_i_pool_output, mode=mode,
                                                        hidden_units=FLAGS.attention_num_units_forward.split(","),
                                                        dropout=FLAGS.drop_out,
                                                        activation_fn=activation,
                                                        dnn_parent_scope='p_fc',
                                                        is_training=is_training,
                                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.dnn_l2_weight),
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                        FLAGS=FLAGS)
                            embedding_i_pool_output = tf.contrib.layers.batch_norm(embedding_i_pool_output,
                                                                           is_training=is_training)

                        with variable_scope.variable_scope('gru', partitioner=None):
                            ubb_embed_id_i_reshape = tf.reshape(ubb_embed_id_i, [tf.shape(user_emb)[0], -1, FLAGS.ubb_dim])
                            ubb_embed_value_i_reshape = tf.reshape(ubb_embed_value_i, [tf.shape(user_emb)[0], -1, 1])

                            clk_jfy_values.append(tf.reduce_sum(ubb_embed_value_i_reshape, axis=1))

                            embedding_i_concat = tf.slice(ubb_embed_id_i_reshape, [0, 0, 0], [-1, FLAGS.ubb_pos_slice_len, -1])
                            embedding_i_concat_counts = tf.slice(ubb_embed_value_i_reshape, [0, 0, 0], [-1, FLAGS.ubb_pos_slice_len, -1])
                            value2gru = tf.slice(value_i, [0, 0, 0], [-1, FLAGS.ubb_pos_slice_len, -1])
                            ubb_seq_length = tf.count_nonzero(tf.squeeze(embedding_i_concat_counts, axis=-1), axis=-1)

                            ###########if have negative Behavior
                            #######################################3
                            # with variable_scope.variable_scope('uic'):
                            #     # embedding_neg_uic = tf.tile(tf.expand_dims(noclk_jfy_emb[i], axis=1), [1, FLAGS.ubb_pos_slice_len, 1])
                            #     with variable_scope.variable_scope('pos', partitioner=None):
                            #         embedding_pos_uic = tf.contrib.layers.fully_connected(inputs=value2gru,
                            #                                                               num_outputs=FLAGS.ubb_pyrimid_fc_unit,
                            #                                                               activation_fn=activation,
                            #                                                               normalizer_fn=None,
                            #                                                               weights_initializer=tf.contrib.layers.xavier_initializer()
                            #                                                               )
                            #         embedding_pos_uic = tf.contrib.layers.batch_norm(embedding_pos_uic,
                            #                                                          is_training=is_training)
                            #     with variable_scope.variable_scope('neg', partitioner=None):
                            #         embedding_neg_uic = tf.contrib.layers.fully_connected(inputs=noclk_jfy_emb[i],
                            #                                                               num_outputs=FLAGS.ubb_pyrimid_fc_unit,
                            #                                                               activation_fn=activation,
                            #                                                               normalizer_fn=None,
                            #                                                               weights_initializer=tf.contrib.layers.xavier_initializer()
                            #                                                               )
                            #         embedding_neg_uic = tf.contrib.layers.batch_norm(embedding_neg_uic,
                            #                                                          is_training=is_training)
                            #         embedding_neg_uic = tf.tile(tf.expand_dims(embedding_neg_uic, axis=1),
                            #                                     [1, FLAGS.ubb_pos_slice_len, 1])
                            #     self._distincelist.append(
                            #         tf.sqrt(tf.reduce_sum(tf.square(embedding_pos_uic - embedding_neg_uic), -1)))
                            #     dis = tf.expand_dims(tf.nn.softmax(
                            #         tf.sqrt(tf.reduce_sum(tf.square(embedding_pos_uic - embedding_neg_uic), -1))), -1)
                            #     embedding_pos_uic_end = tf.multiply(value2gru, dis)
                            #     value2gru = embedding_pos_uic_end

                            cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.attention_num_units)
                            outputs, last_states = tf.nn.dynamic_rnn(
                                cell=cell,
                                inputs=value2gru,
                                sequence_length=ubb_seq_length,
                                dtype=tf.float32)
                            embedding_gru = last_states
                            clk_jfy_pyramid.append(embedding_gru)

                        embedding_pool_att = tf.concat([embedding_i_pool_output, embedding_gru], -1)
                        clk_jfy_emb.append(embedding_pool_att)

                with variable_scope.variable_scope('pyramid', partitioner=None):
                    tensors = []
                    self_att_input_list = clk_jfy_emb
                    for tensori in clk_jfy_emb:
                        tensors.append(tf.expand_dims(tensori, axis=1))
                    input_to_gru = tf.concat(tensors, axis=1)
                    print 'input_gru', input_to_gru

                    seqlength = len(self._ubb_time) * tf.ones_like(ubb_seq_length)
                    outputs, attweight = dnn_multihead_attention(queries=input_to_gru, keys=input_to_gru,
                                                                 key_length=seqlength, scope='pyr',
                                                                 num_heads=4, keep_prob=keep_prob,
                                                                 num_units=FLAGS.attention_num_units,
                                                                 num_output_units=FLAGS.attention_num_output_units,
                                                                 is_training=is_training,
                                                                 num_units_forward=map(int,
                                                                                       FLAGS.attention_num_units_forward.split(
                                                                                           ",")),
                                                                 activation_fn=activation)
                    self._attention_embedding = outputs
                    outputs = tf.reshape(outputs, [tf.shape(item_emb)[0],
                                                   int(FLAGS.attention_num_units_forward.split(",")[-1]) * len(
                                                       self._ubb_time)])
                    clk_jfy_emb.append(outputs)

                with variable_scope.variable_scope('orig', partitioner=None):
                    m0 = clk_jfy_orig_emb[0]
                    for j in range(1, len(self._ubb_time)):
                        with variable_scope.variable_scope(str(j), partitioner=None):
                            m1 = clk_jfy_orig_emb[j]
                            m = tf.concat([m0, m1], axis=1)
                            p_emb = pool(FLAGS, m, name='ubb_orig')
                            clk_jfy_emb.append(p_emb)
                            m0 = m

                clk_jfy_embedding = tf.concat(clk_jfy_emb, -1)

            ubb_embedding = clk_jfy_embedding

            pos_emb = ubb_embedding
        item_emb = item_id_emb
        user_emb = tf.concat([user_emb, pos_emb], axis=-1)


        with variable_scope.variable_scope("semi_Rec"):
            with variable_scope.variable_scope("g_emb"):
                self._rec_loss = 0
                group_user_emb_list = []

                with variable_scope.variable_scope("g_jfy_clk"):
                    attention_emb_list = tf.unstack(self._attention_embedding, axis=1)
                    for i in range(len(self._ubb_time)):
                        user_emb_ori = self_att_input_list[i]
                        user_emb_g = attention_emb_list[i]
                        max_index, group_user_emb, loss_rec = self.group_cluster(scope='g_'+str(i), user_emb_ori=user_emb_ori,
                                                                                  user_emb_g=user_emb_g, FLAGS=FLAGS,
                                                                                  mode=mode, activation=activation,
                                                                                  is_training=is_training, user_emb_num=FLAGS.user_emb_num)
                        self._rec_loss += loss_rec
                        group_user_emb_list.append(group_user_emb)

                self._group_user_emb = tf.concat(group_user_emb_list, -1)

        tf.summary.scalar("group_cluster/rec_loss", tf.reduce_mean(self._rec_loss))

        with variable_scope.variable_scope("traget_att"):
            with variable_scope.variable_scope("user"):
                user_emb_for_att = tf.contrib.layers.fully_connected(inputs=user_emb,
                                                                     num_outputs=FLAGS.embedding_for_att,
                                                                     activation_fn=None,
                                                                     normalizer_fn=None,
                                                                     weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                     )
                user_emb_for_att = tf.contrib.layers.batch_norm(user_emb_for_att, is_training=is_training)
            with variable_scope.variable_scope("group"):
                group_emb_for_att = tf.contrib.layers.fully_connected(inputs=self._group_user_emb,
                                                                      num_outputs=FLAGS.embedding_for_att,
                                                                      activation_fn=None,
                                                                      normalizer_fn=None,
                                                                      weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                      )
                group_emb_for_att = tf.contrib.layers.batch_norm(group_emb_for_att, is_training=is_training)

            with variable_scope.variable_scope("attention"):
                keys = tf.concat([tf.expand_dims(user_emb_for_att, axis=1), tf.expand_dims(group_emb_for_att, axis=1)],
                                 1)
                att = tf.expand_dims(item_emb, 1)
                all_length_ubc = 2 * tf.ones([tf.shape(item_id_emb)[0], ])
                outputs, attweight = multihead_attention(queries=att, queries_length=None,
                                                         keys=keys,
                                                         num_units=FLAGS.attention_num_units,
                                                         num_output_units=FLAGS.embedding_for_att_unit,
                                                         activation_fn=activation, keep_prob=keep_prob,
                                                         keys_length=all_length_ubc, is_training=is_training,
                                                         scope='gatt',
                                                         reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                         num_heads=1)
                outputs = tf.reshape(outputs, [tf.shape(item_emb)[0], 2 * int(FLAGS.attention_num_units)])

        output_emb1 = tf.concat([item_emb, outputs], axis=-1)


        with variable_scope.variable_scope('forward', partitioner=None):
            with variable_scope.variable_scope('ctr', partitioner=None):
                with variable_scope.variable_scope('user', partitioner=None):
                    output_emb1 = dnn_layer(name='user', net=output_emb1, mode=mode,
                                            hidden_units=FLAGS.decision_layer_units.split(","),
                                            dropout=FLAGS.drop_out,
                                            activation_fn=activation,
                                            dnn_parent_scope='user',
                                            is_training=is_training,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                scale=FLAGS.dnn_l2_weight),
                                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                                            , FLAGS=FLAGS)
                    output_emb1 = tf.contrib.layers.batch_norm(output_emb1,
                                                               is_training=is_training)

                with variable_scope.variable_scope("logits") as scope:
                    deep_logits1 = tf.contrib.layers.fully_connected(inputs=output_emb1,
                                                                     num_outputs=1,
                                                                     activation_fn=None,
                                                                     normalizer_fn=None,
                                                                     weights_initializer=tf.contrib.layers.xavier_initializer()
                                                                     )

                    deep_logits = deep_logits1
                    logits = deep_logits
                    self._deep_logits1 = logits

            logits = deep_logits
            print '------', item_emb_to_be, user_emb, '-----------'
            rank_predict = logits
            self._predict_score = tf.sigmoid(logits)

        var_models = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        for v in var_models:
            ops.add_to_collections([ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES], v)

        self._rank_predict = tf.identity(self._predict_score, name="rank_predict")
        self._user_emb = user_emb
        self._deep_logits = deep_logits
        self._logits = logits
        self._item_emb = item_emb


    def opt(self, loss, global_step):

        dnn_learning_rate = self._dnn_lr
        dnn_optimizer = tf.train.AdamOptimizer(learning_rate=dnn_learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            add_summary_gradient(dnn_optimizer, loss, scop_gra=['forward'])
            # gradients = dnn_optimizer.compute_gradients(loss)
            # for idx, itr_g in enumerate(gradients):
            #     variable_summaries("grandients_layer%d" % idx, itr_g[0])
            if self._clip_gradients == 'true':
                print "clip gradients"
                gradients = dnn_optimizer.compute_gradients(loss)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                    grad is not None]

                self._gradients = capped_gradients

                train_op = dnn_optimizer.apply_gradients(capped_gradients, global_step)
            else:
                train_op = dnn_optimizer.minimize(loss, global_step=global_step)
        self._train_op = train_op
        self._dnn_learning_rate = dnn_learning_rate

    def loss(self, features, labels, logits):

        if self._logits_dimension == 1:
            print("sigmoid cross entropy loss")
            raw_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels['click'], tf.float32),
                                                               logits=logits,
                                                               name='logits_cross_entropy')
        else:
            print("softmax cross entropy loss")
            raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels['click'], tf.float32),
                                                                      logits=logits)

        weighted_loss = raw_loss
        ltr_loss = weighted_loss + self._FLAGS.rate_for_recloss*self._rec_loss

        reduce_loss = tf.reduce_mean(ltr_loss)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = reduce_loss + reg_loss
        self._loss = loss

        # use moving average loss to indicate training loss
        ema = tf.train.ExponentialMovingAverage(decay=self._ema_decay, zero_debias=True)
        self._loss_ema_op = ema.apply([reduce_loss, reg_loss, loss])
        self._avg_reduce_loss = ema.average(reduce_loss)
        self._avg_reg_loss = ema.average(reg_loss)
        self._avg_loss = ema.average(loss)

    def metric(self, labels, predict_score):
        _, self._ctr_auc_op = tf.metrics.auc(labels=tf.cast(labels['click'], tf.bool), predictions=predict_score)

        tf.summary.scalar("ori_loss", self._avg_reduce_loss)
        tf.summary.scalar("reg_loss", self._avg_reg_loss)
        tf.summary.scalar("total_loss", self._avg_loss)
        tf.summary.scalar("batch_loss", self._loss)
        variable_summaries("item_emb", self._item_emb)
        variable_summaries("user_emb", self._user_emb)
        variable_summaries("labels", tf.cast(labels['click'], tf.float32))
        variable_summaries("predict_score", predict_score)

    def build(self, mode, features, labels, feature_column_builder, global_step=None, is_training=True,
              eval_item_flag=False, eval_user_flag=False, eval_online2offline=False):

        self._mode = mode
        self._feature_column_builder = feature_column_builder
        self._global_step = global_step
        FLAGS = self._FLAGS
        self._feature_column_builder.buildColumns()

        self.input_layer(self._feature_column_builder)
        self.net(mode, features, is_training)
        self.loss(features, labels, self._logits)
        self.metric(labels, self._predict_score)

        # PREDICT: without labels, just return predict scores.
        # No need for metrics and loss optimization
        if mode == ModeKeys.EVAL:
            return self._logits, self._predict_score, self._ctr_auc_op, self._deep_logits

        self.opt(self._loss, global_step)

        if self._mode == ModeKeys.TRAIN:
            tf.summary.scalar("dnn_learning_rate", self._dnn_learning_rate)

        return self._loss, self._train_op, self._ctr_auc_op, self._loss_ema_op, self._avg_loss, self._logits