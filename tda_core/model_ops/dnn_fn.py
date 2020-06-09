import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.estimator.model_fn import ModeKeys
from attention import multihead_attention, feedforward, multihead_attention_keep, multihead_attention_count
from tensorflow.contrib import layers


DNN_ACTIVATION_FUNCTIONS = {
    "relu": nn.relu,
    "relu6": nn.relu6,
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "leaky_relu": tf.nn.leaky_relu
}


def _add_hidden_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)
    tf.summary.scalar('%s/norm' % tag, tf.norm(value))

def dnn_layer(name, net, mode, hidden_units,
              activation_fn=nn.relu, kernel_initializer=init_ops.glorot_uniform_initializer(),
              kernel_regularizer=None, dropout=None, input_layer_partitioner=None, dnn_parent_scope=None,
              is_training=True,
              FLAGS = None
              ):

    for layer_id, num_hidden_units in enumerate(hidden_units):
        # if (layer_id + 1) == len(hidden_units):
        #     activation_fn = None

        if (layer_id + 1) == len(hidden_units) and FLAGS != None and name == 'seq_to_user_emb':
            num_hidden_units = str(int(num_hidden_units) + FLAGS.item_real_num_unit)
        with variable_scope.variable_scope(
                dnn_parent_scope + 'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
            net = core_layers.dense(
                net,
                units=num_hidden_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=hidden_layer_scope)
            if (layer_id+1)<len(hidden_units):
                net = tf.contrib.layers.batch_norm(net, is_training=is_training)
            # else:
            #     if flag_bn:
            #         net = tf.contrib.layers.batch_norm(net, is_training=is_training)
            # if dropout is not None and mode == ModeKeys.TRAIN:
            #     net = core_layers.dropout(net, rate=dropout, training=True)
            if dropout is not None:
                keep_prob = 1-dropout
                net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
        _add_hidden_layer_summary(net, hidden_layer_scope.name)
    return net


def dnn_layer_with_input_fc(features, mode, hidden_units, feature_columns,
                            activation_fn=nn.relu, kernel_initializer=init_ops.glorot_uniform_initializer(),
                            kernel_regularizer=None, dropout=None, input_layer_partitioner=None, dnn_parent_scope=None, is_training=True):

    with variable_scope.variable_scope(
            dnn_parent_scope+'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner):
        net = tf.contrib.layers.input_from_feature_columns(
            features, feature_columns)
        input = net

        # net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    for layer_id, num_hidden_units in enumerate(hidden_units):
        with variable_scope.variable_scope(
                dnn_parent_scope+'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
            # if (layer_id + 1) == len(hidden_units):
            #     activation_fn = None
            net = core_layers.dense(
                net,
                units=num_hidden_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=hidden_layer_scope)
            if (layer_id+1)<len(hidden_units):
                net = tf.contrib.layers.batch_norm(net, is_training=is_training)
            if dropout is not None and mode == ModeKeys.TRAIN:
                net = core_layers.dropout(net, rate=dropout, training=True)
        _add_hidden_layer_summary(net, hidden_layer_scope.name)
    return net, input

def dnn_logit_seq(features, feature_columns, key_length_column, seq_length, user_profile_emb=None, sequence_item_combiner=None, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu, input_layer_partitioner=None, dnn_parent_scope=None, attention_seq_input_colum='input_layer'
                  , attention_blocks=2,target_attention=None):
    attention_collections_dnn_hidden_layer = "attention_dnn_hidden_layer"
    attention_collections_dnn_hidden_output = "attention_dnn_hidden_output"
    with variable_scope.variable_scope(
            'seq_input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner):
        if attention_seq_input_colum == 'input_layer':
            input_seq = tf.contrib.layers.input_from_feature_columns(
            features, feature_columns)
        elif attention_seq_input_colum == 'input_from_feature_columns':
            input_seq = layers.input_from_feature_columns(features, feature_columns)

        key_length = tf.to_int32(tf.contrib.layers.input_from_feature_columns(features, key_length_column))

        net = input_seq
        # net = tf.contrib.layers.batch_norm(input_seq, is_training=is_training)
    items = tf.split(net, seq_length, axis=1)  # [batch_size, f_num*f_embedding]*length
    items_stack = tf.stack(values=items, axis=1)  # [batch_size, length, f_num*f_embedding] or [h*N, T_q, T_k]
    if sequence_item_combiner == 'mean':
        seq_vec = tf.reduce_mean(items_stack, axis=1)  # (N, d)
    elif sequence_item_combiner == 'max':
        seq_vec = tf.reduce_max(items_stack, axis=1)  # (N, d)
    elif sequence_item_combiner == 'concat':
        seq_vec = tf.reshape(items_stack,
                                      shape=[-1, items_stack.shape[1] * items_stack.shape[2]])  # (N,L*d)
    elif sequence_item_combiner == 'attention':
        print 'self attention model'
        s_item_vec, stt_vec = multihead_attention(queries=items_stack, queries_length=key_length, keys=items_stack,
                                                 num_units=num_units, num_output_units=num_output_units,
                                                 activation_fn=activation_fn, keep_prob=keep_prob,
                                                 keys_length=key_length, is_training=is_training, scope="self_attention",
                                                 reuse=tf.AUTO_REUSE,query_masks=None, key_masks=None,
                                                 variables_collections=[attention_collections_dnn_hidden_layer],
                                                 outputs_collections=[attention_collections_dnn_hidden_output],
                                                 num_heads=num_heads)
        seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="feed_forward",
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

    elif sequence_item_combiner == 'attention_with_user_attention':
        print 'attention_with_user_attention'

        s_item_vec, stt_vec = multihead_attention(queries=items_stack, queries_length=key_length, keys=items_stack,
                                                 num_units=num_units, num_output_units=num_output_units,
                                                 activation_fn=activation_fn, keep_prob=keep_prob,
                                                 keys_length=key_length, is_training=is_training, scope="self_attention",
                                                 reuse=tf.AUTO_REUSE,query_masks=None, key_masks=None,
                                                 variables_collections=[attention_collections_dnn_hidden_layer],
                                                 outputs_collections=[attention_collections_dnn_hidden_output],
                                                 num_heads=num_heads)
        seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="item_feed_forward",
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        if user_profile_emb is None:
            raise Exception('user_profile_emb is None')

        attention = user_profile_emb
        attention = tf.expand_dims(attention, 1)

        user_s_item_vec, user_stt_vec = multihead_attention(queries=attention, queries_length=None,
                                                    keys=seq_vec,
                                                    num_units=num_units, num_output_units=num_output_units,
                                                    activation_fn=activation_fn, keep_prob=keep_prob,
                                                    keys_length=key_length, is_training=is_training, scope="user_layer_attention",
                                                    reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                    variables_collections=[attention_collections_dnn_hidden_layer],
                                                    outputs_collections=[attention_collections_dnn_hidden_output],
                                                    num_heads=num_heads)

        user_profile_seq_vec = feedforward(user_s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="user_feed_forward",
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        seq_vec = user_profile_seq_vec

    elif sequence_item_combiner == 'attention_with_antoint_with_user_attention':
        print 'attention_with_antoint_with_user_attention'
        seq_vec = items_stack
        for i in range(attention_blocks):
            s_item_vec, stt_vec = multihead_attention(queries=seq_vec, queries_length=key_length, keys=seq_vec,
                                                  num_units=num_units, num_output_units=num_output_units,
                                                  activation_fn=activation_fn, keep_prob=keep_prob,
                                                  keys_length=key_length, is_training=is_training,
                                                  scope="self_attention_%s"%(str(i)),
                                                  reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                  variables_collections=[attention_collections_dnn_hidden_layer],
                                                  outputs_collections=[attention_collections_dnn_hidden_output],
                                                  num_heads=num_heads)
            seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="feed_forward_%s"%(str(i)),
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        if user_profile_emb is None:
            raise Exception('user_profile_emb is None')

        attention = user_profile_emb
        attention = tf.expand_dims(attention, 1)

        user_s_item_vec, user_stt_vec = multihead_attention(queries=attention, queries_length=None,
                                                            keys=seq_vec,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope="user_layer_attention",
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

        user_profile_seq_vec = feedforward(user_s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                                           scope="feed_forward",
                                           reuse=tf.AUTO_REUSE,
                                           variables_collections=[attention_collections_dnn_hidden_layer],
                                           outputs_collections=[attention_collections_dnn_hidden_output],
                                           is_training=is_training)

        seq_vec = user_profile_seq_vec

    elif sequence_item_combiner == 'attention_with_user_attention':
        print 'attention_with_user_attention'

        s_item_vec, stt_vec = multihead_attention(queries=items_stack, queries_length=key_length, keys=items_stack,
                                                 num_units=num_units, num_output_units=num_output_units,
                                                 activation_fn=activation_fn, keep_prob=keep_prob,
                                                 keys_length=key_length, is_training=is_training, scope="self_attention",
                                                 reuse=tf.AUTO_REUSE,query_masks=None, key_masks=None,
                                                 variables_collections=[attention_collections_dnn_hidden_layer],
                                                 outputs_collections=[attention_collections_dnn_hidden_output],
                                                 num_heads=num_heads)
        seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="item_feed_forward",
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        if user_profile_emb is None:
            raise Exception('user_profile_emb is None')

        attention = user_profile_emb
        attention = tf.expand_dims(attention, 1)

        user_s_item_vec, user_stt_vec = multihead_attention(queries=attention, queries_length=None,
                                                    keys=seq_vec,
                                                    num_units=num_units, num_output_units=num_output_units,
                                                    activation_fn=activation_fn, keep_prob=keep_prob,
                                                    keys_length=key_length, is_training=is_training, scope="user_layer_attention",
                                                    reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                    variables_collections=[attention_collections_dnn_hidden_layer],
                                                    outputs_collections=[attention_collections_dnn_hidden_output],
                                                    num_heads=num_heads)

        user_profile_seq_vec = feedforward(user_s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="user_feed_forward",
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        seq_vec = user_profile_seq_vec

    elif sequence_item_combiner == 'attention_with_antoint_with_user_attention_targetatt':
        print 'attention_with_antoint_with_user_attention_targetatt'
        seq_vec = items_stack
        for i in range(attention_blocks):
            s_item_vec, stt_vec = multihead_attention(queries=seq_vec, queries_length=key_length, keys=seq_vec,
                                                  num_units=num_units, num_output_units=num_output_units,
                                                  activation_fn=activation_fn, keep_prob=keep_prob,
                                                  keys_length=key_length, is_training=is_training,
                                                  scope="self_attention_%s"%(str(i)),
                                                  reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                  variables_collections=[attention_collections_dnn_hidden_layer],
                                                  outputs_collections=[attention_collections_dnn_hidden_output],
                                                  num_heads=num_heads)
            seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="feed_forward_%s"%(str(i)),
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        target = tf.expand_dims(target_attention, 1)
        target_item_vec, target_stt_vec = multihead_attention(queries=target, queries_length=None,
                                                              keys=seq_vec,
                                                              num_units=num_units, num_output_units=num_output_units,
                                                              activation_fn=activation_fn, keep_prob=keep_prob,
                                                              keys_length=key_length, is_training=is_training,
                                                              scope="target_layer_attention",
                                                              reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                              variables_collections=[
                                                                  attention_collections_dnn_hidden_layer],
                                                              outputs_collections=[
                                                                  attention_collections_dnn_hidden_output],
                                                              num_heads=num_heads)

        target_seq_vec = feedforward(target_item_vec, num_units=num_units_forward,
                                     activation_fn=activation_fn,
                                     scope="target_feed_forward",
                                     reuse=tf.AUTO_REUSE,
                                     variables_collections=[attention_collections_dnn_hidden_layer],
                                     outputs_collections=[attention_collections_dnn_hidden_output],
                                     is_training=is_training)


        if user_profile_emb is None:
            raise Exception('user_profile_emb is None')

        attention = user_profile_emb
        attention = tf.expand_dims(attention, 1)

        user_s_item_vec, user_stt_vec = multihead_attention(queries=attention, queries_length=None,
                                                            keys=target_seq_vec,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope="user_layer_attention",
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

        user_profile_seq_vec = feedforward(user_s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                                           scope="feed_forward",
                                           reuse=tf.AUTO_REUSE,
                                           variables_collections=[attention_collections_dnn_hidden_layer],
                                           outputs_collections=[attention_collections_dnn_hidden_output],
                                           is_training=is_training)



        seq_vec = user_profile_seq_vec

    elif sequence_item_combiner == 'attention_with_antoint_after_user_targetatt':
        print 'attention_with_antoint_after_user_targetatt'
        seq_vec = items_stack
        for i in range(attention_blocks):
            s_item_vec, stt_vec = multihead_attention(queries=seq_vec, queries_length=key_length, keys=seq_vec,
                                                  num_units=num_units, num_output_units=num_output_units,
                                                  activation_fn=activation_fn, keep_prob=keep_prob,
                                                  keys_length=key_length, is_training=is_training,
                                                  scope="self_attention_%s"%(str(i)),
                                                  reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                  variables_collections=[attention_collections_dnn_hidden_layer],
                                                  outputs_collections=[attention_collections_dnn_hidden_output],
                                                  num_heads=num_heads)
            seq_vec = feedforward(s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                              scope="feed_forward_%s"%(str(i)),
                              reuse=tf.AUTO_REUSE, variables_collections=[attention_collections_dnn_hidden_layer],
                              outputs_collections=[attention_collections_dnn_hidden_output], is_training=is_training)

        if user_profile_emb is None:
            raise Exception('user_profile_emb is None')

        attention = user_profile_emb
        attention = tf.expand_dims(attention, 1)

        user_s_item_vec, user_stt_vec = multihead_attention(queries=attention, queries_length=None,
                                                            keys=seq_vec,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope="user_layer_attention",
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

        user_profile_seq_vec = feedforward(user_s_item_vec, num_units=num_units_forward, activation_fn=activation_fn,
                                           scope="feed_forward",
                                           reuse=tf.AUTO_REUSE,
                                           variables_collections=[attention_collections_dnn_hidden_layer],
                                           outputs_collections=[attention_collections_dnn_hidden_output],
                                           is_training=is_training)

        target = tf.expand_dims(target_attention, 1)
        target_item_vec, target_stt_vec = multihead_attention(queries=target, queries_length=None,
                                                              keys=user_profile_seq_vec,
                                                              num_units=num_units, num_output_units=num_output_units,
                                                              activation_fn=activation_fn, keep_prob=keep_prob,
                                                              keys_length=key_length, is_training=is_training,
                                                              scope="target_layer_attention",
                                                              reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                              variables_collections=[
                                                                  attention_collections_dnn_hidden_layer],
                                                              outputs_collections=[
                                                                  attention_collections_dnn_hidden_output],
                                                              num_heads=num_heads)

        target_seq_vec = feedforward(target_item_vec, num_units=num_units_forward,
                                     activation_fn=activation_fn,
                                     scope="target_feed_forward",
                                     reuse=tf.AUTO_REUSE,
                                     variables_collections=[attention_collections_dnn_hidden_layer],
                                     outputs_collections=[attention_collections_dnn_hidden_output],
                                     is_training=is_training)

        seq_vec = target_seq_vec

    else:
        seq_vec = tf.reshape(items_stack,
                             shape=[-1, items_stack.shape[1] * items_stack.shape[2]])  # (N,L*d)
    return seq_vec, input_seq


def dnn_multihead_attention(queries, keys, key_length, scope, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu):
    attention_collections_dnn_hidden_layer = scope+'att_in'
    attention_collections_dnn_hidden_output = scope+'att_out'
    with variable_scope.variable_scope(
            scope) as scope1:
        vec1, vec2 = multihead_attention(queries=queries, queries_length=None,
                                                            keys=keys,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope=scope,
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)
        if num_units_forward:
            seq_vec = feedforward(vec1, num_units=num_units_forward, activation_fn=activation_fn,
                                               scope=scope+"forward",
                                               reuse=tf.AUTO_REUSE,
                                               variables_collections=[attention_collections_dnn_hidden_layer],
                                               outputs_collections=[attention_collections_dnn_hidden_output],
                                               is_training=is_training)
        else:
            seq_vec = vec1
    return seq_vec, vec2

def keep_multihead_attention(queries, keys, key_length, scope, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu):
    attention_collections_dnn_hidden_layer = scope+'att_in'
    attention_collections_dnn_hidden_output = scope+'att_out'
    with variable_scope.variable_scope(
            scope) as scope1:
        vec1, vec2 = multihead_attention_keep(queries=queries, queries_length=None,
                                                            keys=keys,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope=scope,
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

        seq_vec = feedforward(vec1, num_units=num_units_forward, activation_fn=activation_fn,
                                           scope=scope+"forward",
                                           reuse=tf.AUTO_REUSE,
                                           variables_collections=[attention_collections_dnn_hidden_layer],
                                           outputs_collections=[attention_collections_dnn_hidden_output],
                                           is_training=is_training)

    return seq_vec, vec2


def dnn_multihead_attention_count(queries, keys, key_length, scope, counts, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu):
    attention_collections_dnn_hidden_layer = scope+'att_in'
    attention_collections_dnn_hidden_output = scope+'att_out'
    with variable_scope.variable_scope(
            scope) as scope1:
        vec1, vec2 = multihead_attention_count(queries=queries, queries_length=None,
                                                            keys=keys,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, counts=counts,
                                                            is_training=is_training, scope=scope,
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

        seq_vec = feedforward(vec1, num_units=num_units_forward, activation_fn=activation_fn,
                                           scope=scope+"forward",
                                           reuse=tf.AUTO_REUSE,
                                           variables_collections=[attention_collections_dnn_hidden_layer],
                                           outputs_collections=[attention_collections_dnn_hidden_output],
                                           is_training=is_training)
    return seq_vec, vec2


def dnn_multihead_attention_count_without_forward(queries, keys, key_length, scope, counts, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu):
    attention_collections_dnn_hidden_layer = scope+'att_in'
    attention_collections_dnn_hidden_output = scope+'att_out'
    with variable_scope.variable_scope(
            scope) as scope1:
        vec1, vec2 = multihead_attention_count(queries=queries, queries_length=None,
                                                            keys=keys,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, counts=counts,
                                                            is_training=is_training, scope=scope,
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

    return vec1, vec2


def keep_multihead_attention_without_forward(queries, keys, key_length, scope, num_heads=4,
                 keep_prob=None, num_units=16, num_output_units=32, is_training=True, num_units_forward=[16, 8],
                 activation_fn=nn.relu):
    attention_collections_dnn_hidden_layer = scope+'att_in'
    attention_collections_dnn_hidden_output = scope+'att_out'
    with variable_scope.variable_scope(
            scope) as scope1:
        vec1, vec2 = multihead_attention_keep(queries=queries, queries_length=None,
                                                            keys=keys,
                                                            num_units=num_units, num_output_units=num_output_units,
                                                            activation_fn=activation_fn, keep_prob=keep_prob,
                                                            keys_length=key_length, is_training=is_training,
                                                            scope=scope,
                                                            reuse=tf.AUTO_REUSE, query_masks=None, key_masks=None,
                                                            variables_collections=[
                                                                attention_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                attention_collections_dnn_hidden_output],
                                                            num_heads=num_heads)

    return vec1, vec2