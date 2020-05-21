import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, Activation, Add, Lambda, Multiply, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import Constant
# from keras.engine.topology import Layer
from tensorflow.keras import backend as K

class Factorization_machine(Layer):

    def __init__(self, k, **kwargs):
        self.k = k
        super(Factorization_machine, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.w0 = self.add_weight(initializer=tf.constant_initializer([0]), shape=[1], dtype=tf.float32, trainable=True, name='fm_w0')
        self.w = self.add_weight(initializer=tf.constant_initializer([0]), shape=[input_shape[-1]], dtype=tf.float32, trainable=True, name='fm_w')
        self.V = self.add_weight(initializer=tf.constant_initializer([0]), shape=[self.k, input_shape[-1]], dtype=tf.float32, trainable=True, name='fm_V')

        super(Factorization_machine, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_vector):
        fm_linear_terms = self.w0 +  tf.matmul(input_vector, tf.expand_dims(self.w, 1))

        fm_interactions_part1 = tf.matmul(input_vector, tf.transpose(self.V))
        fm_interactions_part1 = tf.pow(fm_interactions_part1, 2)

        fm_interactions_part2 = tf.matmul(tf.pow(input_vector, 2), tf.transpose(tf.pow(self.V, 2)))

        fm_interactions = fm_interactions_part1 - fm_interactions_part2

        fm_interactions = tf.reduce_sum(fm_interactions, 2, keepdims = True)

        fm_interactions = tf.multiply(0.5, fm_interactions)

        fm_prediction = tf.add(fm_linear_terms, fm_interactions)

        return fm_prediction



class BAC(Layer):

    def __init__(self, activation = 'softmax', nn_units = 300, emb_dim = 600, **kwargs):
        self.activation = activation
        self.nn_units = nn_units
        self.emb_dim = emb_dim
        super(BAC, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.dense_1 = Dense(self.nn_units, activation=relu, use_bias=True)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(self.nn_units, activation=relu, use_bias=True)
        self.dense_2.build(input_shape)
        self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(BAC, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, stack_input):
        unstack_input = tf.unstack(stack_input)
        passage_input = unstack_input[0]
        query_input = unstack_input[1]
        
        passage_dense = self.dense_1(passage_input)
        query_dense = self.dense_2(query_input)
        affinity_matrix = tf.matmul(passage_dense, tf.transpose(query_dense, perm = [0, 2, 1]))
        affinity_matrix = 1/np.sqrt(self.emb_dim) * affinity_matrix
        activation = Activation(softmax)
        aligned_p = activation(affinity_matrix)
        aligned_q = activation(tf.transpose(affinity_matrix, perm = [0, 2, 1]))
        passage_aligned = tf.matmul(aligned_p, passage_input)
        query_aligned = tf.matmul(aligned_q, query_input)

        passage_concat = tf.concat([query_aligned, passage_input], 2)
        query_concat = tf.concat([passage_aligned, query_input], 2)

        passage_diff = tf.subtract(query_aligned, passage_input)
        query_diff = tf.subtract(passage_aligned, query_input)

        passage_mul = tf.multiply(query_aligned, passage_input)
        query_mul = tf.multiply(passage_aligned, query_input)

        fm_1 = Factorization_machine(5, name='passage_concat_layer')
        fm_2 = Factorization_machine(5, name='query_concat_layer')
        fm_3 = Factorization_machine(5, name='passage_diff_layer')
        fm_4 = Factorization_machine(5, name='query_diff_layer')
        fm_5 = Factorization_machine(5, name='passage_mul_layer')
        fm_6 = Factorization_machine(5, name='query_mul_layer')
        
        connecter_1 = fm_1(passage_concat)
        connecter_2 = fm_2(query_concat)
        connecter_3 = fm_3(passage_diff)
        connecter_4 = fm_4(query_diff)
        connecter_5 = fm_5(passage_mul)
        connecter_6 = fm_6(query_mul)

        feature_p = [connecter_1, connecter_3, connecter_5]
        feature_q = [connecter_2, connecter_4, connecter_6]

        features_passage = tf.concat(feature_p, 2)
        features_query = tf.concat(feature_q, 2)
        print(features_passage.shape)
        return features_passage, features_query


class Highway(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        input_shape_dense_1 = input_shape[-1]
        self.dense_1 = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=dim)
        self.dense_2.build(input_shape)
        self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config




def build_bilstm(emb_dim, max_passage_length = None, max_query_length = None, num_highway_layer = 2, num_decoder = 1, encoder_dropout = 0, decoder_dropout = 0):

    passage_input = Input(shape = (max_passage_length, emb_dim), dtype = 'float32', name = 'passage_input')
    query_input = Input(shape = (max_query_length, emb_dim), dtype = 'float32', name = 'query_input')

    passage_embedding = passage_input
    query_embedding = query_input

    for i in range(num_highway_layer):
        highway_layer = Highway(name = 'highway_{}'.format(i))
        query_layer = TimeDistributed(highway_layer, name = highway_layer.name + '_qtd')
        query_embedding = query_layer(query_embedding)
        passage_layer = TimeDistributed(highway_layer, name = highway_layer.name + '_ptd')
        passage_embedding = passage_layer(passage_embedding)
    
    encoder_layer_p = Bidirectional(LSTM(emb_dim, recurrent_dropout = encoder_dropout, return_sequences = True), name = 'bidirectional_encoder_p')
    encoder_layer_q = Bidirectional(LSTM(emb_dim, recurrent_dropout = encoder_dropout, return_sequences = True), name = 'bidirectional_encoder_q')

    passage_encoded = encoder_layer_p(passage_embedding)
    query_encoded = encoder_layer_q(query_embedding)

    print(passage_embedding.shape, query_embedding.shape)
    print('----------------------------------------------')
    print(passage_encoded.shape, query_encoded.shape)

    connecters = []
    num_lstm_layers = 3
    
    for i in range(num_lstm_layers):

        deca_encoder_p = Bidirectional(LSTM(emb_dim, return_sequences=True), name='deca_encoder_p_{}'.format(i))
        deca_encoder_q = Bidirectional(LSTM(emb_dim, return_sequences=True), name='deca_encoder_q_{}'.format(i))
        bac_layer = BAC(name='bac_layer_{}'.format(i))
        # bac_td = TimeDistributed(bac_layer, name=bac_layer.name + '_td')
        

        passage_deca = deca_encoder_p(passage_encoded)
        query_deca = deca_encoder_q(query_encoded)

        passage_encoded = passage_deca
        query_encoded = query_deca

        connecter_p, connecter_q = bac_layer(tf.stack([passage_deca, query_deca]))

        print(connecter_p.shape, connecter_q.shape)

        connecters.append([connecter_p, connecter_q])


    model = Model(inputs = [passage_input, query_input], outputs = connecters)
    model.build(input_shape=(max_passage_length, emb_dim))
    model.summary()


build_bilstm(600, 200, 200)