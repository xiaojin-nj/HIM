import tensorflow as tf
from tensorflow.contrib import layers

def _add_hidden_layer_summary(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.scalar('norm/' + name, tf.norm(var))
    tf.summary.scalar('fraction_of_zero_values/' + name, tf.nn.zero_fraction(var))
    tf.summary.histogram('histogram/' + name, var)

def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        keep_prob=0.8,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    num_output_units: A scalar. Output Value size.
    keep_prob: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
    query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
    key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q"
                               )  # (N, T_q, C)
    _add_hidden_layer_summary(Q, "Q")
    K = layers.fully_connected(keys,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(K, "K")
    V = layers.fully_connected(keys,
                               num_output_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="V"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(V, "V")

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
    # [batch_size, num_heads, query_len, key_len]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, key_len)  # (N, T_k)
    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    if query_masks is None:
      if queries_length is not None:
        query_masks = tf.sequence_mask(queries_length, query_len)  # (N, T_q)

    if query_masks is not None:
      query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                            [1, num_heads, 1, key_len])
      paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
      outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    def transpose_then_concat_last_two_dimenstion(tensor):
      tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
      t_shape = tensor.get_shape().as_list()
      num_heads, dim = t_shape[-2:]
      return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs, att_vec


def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=None,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                is_training=True):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    outputs = layers.fully_connected(inputs,
                                     num_units[0],
                                     activation_fn=activation_fn,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections,
                                     )
    outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, scope='feed_0_bn')
    outputs = layers.fully_connected(outputs,
                                     num_units[1],
                                     activation_fn=None,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, scope='feed_1_bn')
    outputs += inputs

  return outputs


def multihead_attention_keep(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        keep_prob=0.8,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    num_output_units: A scalar. Output Value size.
    keep_prob: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
    query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
    key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q"
                               )  # (N, T_q, C)
    _add_hidden_layer_summary(Q, "Q")
    K = layers.fully_connected(keys,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(K, "K")
    V = layers.fully_connected(keys,
                               num_output_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="V"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(V, "V")

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
    # [batch_size, num_heads, query_len, key_len]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, key_len)  # (N, T_k)
    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    if query_masks is None:
      if queries_length is not None:
        query_masks = tf.sequence_mask(queries_length, query_len)  # (N, T_q)

    if query_masks is not None:
      query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                            [1, num_heads, 1, key_len])
      paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
      outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.transpose(outputs, [0, 1, 3, 2])
    outputs = tf.multiply(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    def transpose_then_concat_last_two_dimenstion(tensor):
      tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
      t_shape = tensor.get_shape().as_list()
      num_heads, dim = t_shape[-2:]
      return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)
    print outputs,'---'

    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs, att_vec


def multihead_attention_count(queries,
                        queries_length,
                        keys,
                        keys_length,
                        counts,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        keep_prob=0.8,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    num_output_units: A scalar. Output Value size.
    keep_prob: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
    query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
    key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = layers.fully_connected(queries,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="Q"
                               )  # (N, T_q, C)
    _add_hidden_layer_summary(Q, "Q")
    K = layers.fully_connected(keys,
                               num_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="K"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(K, "K")
    V = layers.fully_connected(keys,
                               num_output_units,
                               activation_fn=activation_fn,
                               variables_collections=variables_collections,
                               outputs_collections=outputs_collections, scope="V"
                               )  # (N, T_k, C)
    _add_hidden_layer_summary(V, "V")

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
    # [batch_size, num_heads, query_len, key_len]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    # Key Masking
    if key_masks is None:
      key_masks = tf.sequence_mask(keys_length, key_len)  # (N, T_k)
    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    if query_masks is None:
      if queries_length is not None:
        query_masks = tf.sequence_mask(queries_length, query_len)  # (N, T_q)

    if query_masks is not None:
      query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                            [1, num_heads, 1, key_len])
      paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
      outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    counts = tf.expand_dims(counts, 1)
    counts = tf.tile(counts, [1, num_heads, 1, 1])
    outputs = tf.multiply(outputs, counts)
    att_vec = outputs

    # Dropouts
    outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    def transpose_then_concat_last_two_dimenstion(tensor):
      tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
      t_shape = tensor.get_shape().as_list()
      num_heads, dim = t_shape[-2:]
      return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs, att_vec


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

