import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Activation, LSTM
from tensorflow.keras.models import Model


# class BAC(Layer):

#     def __init__(self, activation = 'softmax', nn_units = 300, **kwargs):
#         self.activation = activation
#         self.nn_units = nn_units
#         super(BAC, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         self.dense_1 = Dense(self.nn_units, activation=relu, use_bias=True, name = 'co_atten_passgae_dense')
#         self.dense_1.build(input_shape)
#         self.dense_2 = Dense(self.nn_units, activation=relu, use_bias=True, name = 'co_atten_query_dense')
#         self.dense_2.build(input_shape)
#         self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights
    
#     def call(self, query_input, passage_input):
#         passage_dense = self.dense_1(passage_input)
#         query_dense = self.dense_2(query_input)
#         affinity_matrix = tf.matmul(passage_dense, tf.transpose(query_dense, perm = [0, 2, 1]))
#         affinity_matrix = 1/np.sqrt(emb_dim) * affinity_matrix
#         activation = Activation(softmax)
#         aligned_p = activation(affinity_matrix)
#         aligned_q = activation(tf.transpose(affinity_matrix, perm = [0, 2, 1]))
#         passage_aligned = tf.matmul(aligned_p, passage_input)
#         query_aligned = tf.matmul(aligned_q, query_input)

#         return passage_aligned, query_aligned


def build_fm(input_vec, k=5):
    
    # seq_lens = tf.shape(input_vec)[1]
    # dims = input_vec.get_shape().as_list()[2]
    # input_vec = tf.reshape(input_vec, [-1, dims])
    print(input_vec.shape)

    fm_k = k
    fm_p = input_vec.get_shape().as_list()[2]
    
    fm_w0 = tf.Variable(tf.constant_initializer([0])(shape=[1], dtype=tf.float32), trainable=True)
    fm_w = tf.Variable(tf.constant_initializer([0])(shape=[fm_p], dtype=tf.float32), trainable=True)

    fm_V = tf.Variable(tf.random_normal_initializer(0, 0.1)([fm_k, fm_p], tf.float32))
    
    fm_linear_terms = fm_w0 +  tf.matmul(input_vec, tf.expand_dims(fm_w, 1))

    fm_interactions_part1 = tf.matmul(input_vec, tf.transpose(fm_V))
    fm_interactions_part1 = tf.pow(fm_interactions_part1, 2)

    fm_interactions_part2 = tf.matmul(tf.pow(input_vec, 2), tf.transpose(tf.pow(fm_V, 2)))


    fm_interactions = fm_interactions_part1 - fm_interactions_part2

    # latent_dim = fm_interactions

    fm_interactions = tf.reduce_sum(fm_interactions, 2, keepdims = True)

    fm_interactions = tf.multiply(0.5, fm_interactions)


    fm_prediction = tf.add(fm_linear_terms, fm_interactions)

    # fm_prediction = tf.reshape(fm_prediction, [-1, seq_lens, 1])
    # latent_dim = tf.reshape(latent_dim, [-1, seq_lens, k])

    return fm_prediction



def co_attention(max_passage_len, max_query_len, emb_dim):

    passage_input = Input(shape = (max_passage_len, emb_dim), dtype = 'float32', name = 'passage_input')
    query_input = Input(shape = (max_query_len, emb_dim), dtype = 'float32', name = 'query_input')

    # print(passage_input.shape, query_input.shape)

    dense_1 = Dense(300, activation=relu, use_bias=True, name = 'Dense_layer')

    passage_affinity = dense_1(passage_input)
    query_affinity = dense_1(query_input)

    # print(passage_affinity.shape, query_affinity.shape)

    affinity_matrix = tf.matmul(passage_affinity, tf.transpose(query_affinity, perm = [0, 2, 1]))

    affinity_matrix = 1/np.sqrt(emb_dim) * affinity_matrix

    # print(affinity_matrix.shape)

    activation = Activation(softmax)
    
    aligned_p = activation(affinity_matrix)
    aligned_q = activation(tf.transpose(affinity_matrix, perm = [0, 2, 1]))

    # print(aligned_p.shape, aligned_q.shape)

    passage_aligned = tf.matmul(aligned_p, passage_input)
    query_aligned = tf.matmul(aligned_q, query_input)

    # print(passage_aligned.shape, query_aligned.shape)

    passage_concat = tf.concat([query_aligned, passage_input], 2)

    query_concat = tf.concat([passage_aligned, query_input], 2)

    passage_diff = tf.subtract(query_aligned, passage_input)
    query_diff = tf.subtract(passage_aligned, query_input)

    passage_mul = tf.multiply(query_aligned, passage_input)
    query_mul = tf.multiply(passage_aligned, query_input)

    connecter_1 = build_fm(passage_concat, 5)
    connecter_2 = build_fm(query_concat, 5)
    connecter_3 = build_fm(passage_diff, 5)
    connecter_4 = build_fm(query_diff, 5)
    connecter_5 = build_fm(passage_mul, 5)
    connecter_6 = build_fm(query_mul, 5)

    connecter = [connecter_1, connecter_2, connecter_3, connecter_4, connecter_5, connecter_6]

    model = Model(inputs = [passage_input, query_input], outputs = [connecter])

    model.summary()


co_attention(200, 200, 600)



encoder_layer = Bidirectional(LSTM(emb_dim, recurrent_dropout = encoder_dropout, return_sequences = True), name = 'bidirectional_encoder')

    passage_encoded = encoder_layer(passage_embedding)
    query_encoded = encoder_layer(query_embedding)

    deca_encoder_layer = Bidirectional(LSTM(emb_dim, recurrent_dropout = encoder_dropout, return_sequences = True), name = 'decaEncoder_layer')

    passage_deca = deca_encoder_layer(passage_encoded)
    query_deca = deca_encoder_layer(query_encoded)

    bac_layer = BAC(name='BAC_layer')
    some_layer = TimeDistributed(bac_layer, name='connecter_Layer')
    passage_connecters, query_connecters = some_layer(passage_deca, query_deca)





    # class Gated_attention(Layer):

#     def __init__(self, num_units = 300, emb_dim = 600, **kwargs):
#         self.num_units = num_units
#         self.emb_dim = emb_dim
#         super(Gated_attention, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         self.dense_1 = Dense(self.num_units, activation=relu)
#         self.dense_2 = Dense(self.num_units, activation=relu)
#         self.dense_3 = Dense(self.num_units, activation=sigmoid)
#         self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights + self.dense_3.trainable_weights

#         super(Gated_attention, self).build(input_shape)
    
#     def call(self, stacked_input):
#         unstacked_input = tf.unstack(stacked_input)
#         input_1 = unstacked_input[0]        # Passage Input
#         input_2 = unstacked_input[1]        # Query Input (P' in case of Self Attention)
        
#         dense_1_op = self.dense_1(input_1)
#         dense_2_op = self.dense_2(input_2)

#         co_mat = tf.matmul(dense_1_op, tf.transpose(dense_2_op, perm = [0, 2, 1]))
#         co_mat = 1/tf.sqrt(self.emb_dim) * co_mat     # TODO: check if emb_dim or input_passage_dim

#         activation = Activation(softmax)
#         passage_bar = activation(co_mat)

#         passage_bar = tf.matmul(passage_bar, input_2)
#         passage_concat = tf.concat([input_1, passage_bar])