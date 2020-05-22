import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, Activation, Add, Lambda, Multiply, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
# from keras.engine.topology import Layer
from tensorflow.keras import backend as K


class Gated_attention_with_self(Layer):

    def __init__(self, passage_len = 200, num_units = 300, emb_dim = 600, **kwargs):
        self.num_units = num_units
        self.emb_dim = emb_dim
        self.passaage_len = passage_len
        super(Gated_attention_with_self, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        passage_shape = (input_shape[0], self.passaage_len, input_shape[-1])
        query_shape = (input_shape[0], input_shape[1] - self.passaage_len, input_shape[-1])
        concat_shape = (input_shape[0], self.passaage_len, input_shape[-1]*2)
        self.dense_1 = Dense(self.num_units, activation=relu)
        self.dense_1.build(passage_shape)
        self.dense_2 = Dense(self.num_units, activation=relu)
        self.dense_2.build(query_shape)
        self.dense_3 = Dense(input_shape[-1], activation=sigmoid)
        self.dense_3.build(concat_shape)
        self.dense_4 = Dense(input_shape[-1], activation=sigmoid)
        self.dense_4.build((input_shape[0], self.passaage_len, input_shape[-1]+(self.emb_dim*2)))
        self.bilstm_1 = Bidirectional(LSTM(self.emb_dim, return_sequences=True))
        self.bilstm_1.build(passage_shape)
        self.bilstm_2 = Bidirectional(LSTM(self.emb_dim, return_sequences=True))
        self.bilstm_2.build(passage_shape)
        self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights + self.dense_3.trainable_weights + self.dense_4.trainable_weights

        super(Gated_attention_with_self, self).build(input_shape)

    def call(self, stacked_input):
        # unstacked_input = tf.unstack(stacked_input)
        input_1 = stacked_input[:, :self.passaage_len, :]        # Passage Input
        input_2 = stacked_input[:, self.passaage_len:, :]        # Query Input (P' in case of Self Attention)

        print(input_1.shape, input_2.shape)
    
        dense_1_op = self.dense_1(input_1)
        dense_2_op = self.dense_2(input_2)

        co_mat = tf.matmul(dense_1_op, tf.transpose(dense_2_op, perm = [0, 2, 1]))
        co_mat = 1/np.sqrt(self.emb_dim) * co_mat     # TODO: check if emb_dim or input_passage_dim

        activation = Activation(softmax)
        passage_bar = activation(co_mat)

        passage_bar = tf.matmul(passage_bar, input_2)
        passage_concat = tf.concat([input_1, passage_bar], 2)

        print(passage_concat.shape)

        dense_3_op = self.dense_3(passage_concat)
        passage_mul = tf.multiply(dense_3_op, input_1)
        
        query_depend_passage = self.bilstm_1(passage_mul)

        # Self Attention Part

        p_dash_concat = tf.concat([input_1, query_depend_passage], 2)
        print(query_depend_passage.shape, p_dash_concat.shape)
        dense_4_op = self.dense_4(p_dash_concat)
        p_dash_mul = tf.multiply(dense_4_op, input_1)
        query_depend_p_dash = self.bilstm_2(p_dash_mul)

        return query_depend_passage, query_depend_p_dash
        


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

    def __init__(self, passage_len = 200, activation = 'softmax', nn_units = 300, emb_dim = 600, **kwargs):
        self.activation = activation
        self.nn_units = nn_units
        self.emb_dim = emb_dim
        self.passage_len = passage_len
        super(BAC, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.dense_1 = Dense(self.nn_units, activation=relu, use_bias=True)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(self.nn_units, activation=relu, use_bias=True)
        self.dense_2.build(input_shape)
        self.trainable_weight = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(BAC, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, stack_input):
        # unstack_input = tf.unstack(stack_input)

        passage_input = stack_input[:, :self.passage_len, :]
        query_input = stack_input[:, self.passage_len:, :]
        
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




def build_bilstm(start_pointer, end_pointer, emb_dim, max_passage_length = None, max_query_length = None, num_highway_layer = 2, num_decoder = 1, encoder_dropout = 0, decoder_dropout = 0):

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

    passage_outs = []
    query_outs = []
    num_lstm_layers = 3
    
    for i in range(num_lstm_layers):

        deca_encoder_p = Bidirectional(LSTM(emb_dim, return_sequences=True), name='deca_encoder_p_{}'.format(i))
        deca_encoder_q = Bidirectional(LSTM(emb_dim, return_sequences=True), name='deca_encoder_q_{}'.format(i))
        
        passage_deca = deca_encoder_p(passage_encoded)
        query_deca = deca_encoder_q(query_encoded)

        passage_outs.append(passage_deca)
        query_outs.append(query_deca)
    
    passage_c = tf.concat(passage_outs, 2)
    query_c = tf.concat(query_outs, 2)
    
    passage_connecters = []
    query_connecters = []

    for i in range(num_lstm_layers):
        for j in range(num_lstm_layers):
            bac_layer = BAC(name='bac_layer_p_{}_q_{}'.format(i, j))
            connecter_p, connecter_q = bac_layer(tf.concat([passage_outs[i], query_outs[j]], 1))

            passage_connecters.append(connecter_p)
            query_connecters.append(connecter_q)

    connecter_pc = tf.concat(passage_connecters, 2)
    connecter_qc = tf.concat(query_connecters, 2)

    passage_decaout = tf.concat([passage_c, connecter_pc], 2)
    query_decaout = tf.concat([query_c, connecter_qc], 2)

    print(passage_decaout.shape, query_decaout.shape)

    gated_attention_layer = Gated_attention_with_self()
    gated_atten_op, gated_self_atten_op = gated_attention_layer(tf.concat([passage_decaout, query_decaout], 1))
    
    atten_outs = [gated_atten_op, gated_self_atten_op]

    conn_list = []

    print(atten_outs[0].shape, query_outs[0].shape)

    for i in range(2):
        for j in range(num_lstm_layers):
            one_sided_bac = BAC(name='one_sided_bac_u{}_q{}'.format(i, j))
            connecter_1, connecter_2 = one_sided_bac(tf.concat([atten_outs[i], query_outs[j]], 1))

            conn_list.append(connecter_1)
            conn_list.append(connecter_2)
    
    conn_c = tf.concat(conn_list, 2)

    m_mat = tf.concat([gated_self_atten_op, conn_c], 2)

    start_projection = Bidirectional(LSTM(emb_dim, name='bilstm_span_start', return_sequences=True))
    end_projection = Bidirectional(LSTM(emb_dim, name='bilstm_span_end', return_sequences=True))

    softamx_start = Dense(1, activation=softmax, use_bias=False, name='start_softmax')
    softamx_end = Dense(1, activation=softmax, use_bias=False, name='end_softmax')

    span_start = start_projection(m_mat)
    span_end = end_projection(span_start)

    start_pred = softamx_start(span_start)
    end_pred = softamx_end(span_end)

    # start_pointer = tf.zeros((32, 200, 1))
    # end_pointer = tf.zeros((32, 200, 1))

    # loss_start = tf.math.log(tf.reduce_sum(tf.multiply(start_pred, start_pointer)))
    # loss_end = tf.math.log(tf.reduce_sum(tf.multiply(end_pred, end_pointer)))

    loss_start = tf.math.log(tf.reduce_sum(tf.multiply(start_pred, start_pred), axis=1))
    loss_end = tf.math.log(tf.reduce_sum(tf.multiply(end_pred, end_pred), axis=1))
    
    cost = tf.reduce_mean(tf.math.add(loss_start, loss_end), axis=0)

    print(loss_start.shape, loss_end.shape, cost.shape)

    model = Model(inputs = [passage_input, query_input], outputs = [start_pred, end_pred])
    model.build(input_shape=(max_passage_length, emb_dim))
    model.summary()

    # model.compile(optimizer=Adam(learning_rate=0.001), loss=cost)



build_bilstm(None, None, 600, 200, 200)