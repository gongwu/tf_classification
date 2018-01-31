# coding: utf8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils


class LSTMModel(object):

    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS
        self.config = config
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.seq_len = config.max_sent_len
        self.word_len = config.max_word_len
        self.word_embed_size = config.word_dim
        self.char_embed_size = config.char_dim
        self.num_class = config.num_class
        self.num_vocab = FLAGS.num_vocab
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.filter_sizes = [1, 2, 3, 4]
        self.num_filters = FLAGS.num_filters
        self.char_lstm_size = 50
        self.lstm_size = 512
        self.mlp_h1_size = 200
        self.layer_size = FLAGS.layer_size
        self.with_char = FLAGS.with_char
        self.char_type = FLAGS.char_type
        self.with_ner = FLAGS.with_ner
        self.with_pos = FLAGS.with_pos
        self.with_rf = FLAGS.with_rf
        self.with_pun = FLAGS.with_pun
        self.with_senti = FLAGS.with_senti
        self.with_attention = FLAGS.with_attention
        self.with_cnn = FLAGS.with_cnn
        self.with_cnn_lstm = FLAGS.with_cnn_lstm
        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.drop_hidden1 = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Add PlaceHolder
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sent_len]
        self.input_x_len = tf.placeholder(tf.int32, (None,))  # [batch_len]
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))  # [batch_size, label_size]
        # Add Word Embedding
        self.we = tf.Variable(FLAGS.we, name='emb')
        if self.with_ner:
            self.input_x_ner = tf.placeholder(tf.int32, (None, self.seq_len))
            self.ner_we = tf.Variable(FLAGS.ner_we, name='ner_emb')
        if self.with_pos:
            self.input_x_pos = tf.placeholder(tf.int32, (None, self.seq_len))
            self.pos_we = tf.Variable(FLAGS.pos_we, name='pos_emb')
        if self.with_rf:
            self.input_rf = tf.placeholder(tf.float32, (None, self.num_vocab))
        if self.with_pun:
            self.input_x_pun = tf.placeholder(tf.float32, (None, 9))
        if self.with_senti:
            self.input_x_senti = tf.placeholder(tf.float32, (None, 110))
        if self.with_char:
            # [batch_size, sent_len, word_len]
            self.input_x_char = tf.placeholder(tf.int32, (None, self.seq_len, self.word_len))
            self.input_x_char_len = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sen_len]
            # The Char Embedding is Random Initialization
            self.char_we = tf.Variable(FLAGS.char_we, name='char_emb')


        # attention process:
        # 1.get logits for each word in the sentence.
        # 2.get possibility distribution for each word in the sentence.
        # 3.get weighted sum for the sentence as sentence representation.
        def attention_word_level(hidden_state, hidden_size, sequence_length, seq_len, scope=None, reuse=None):
            """
            hidden_state: [batch_size, sequence_length, hidden_size*2]
            context vector:
            :return [batch_size*num_sentences, hidden_size*2]
            """
            with tf.variable_scope(scope or "attention", reuse=reuse):
                self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                          shape=[hidden_size*2, hidden_size*2])
                self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[hidden_size*2])
                self.context_vecotor_word = tf.get_variable("what_is_the_informative_word",
                                                            shape=[hidden_size*2])  # TODO o.k to use batch_size in first demension?
                # 0) one layer of feed forward network
                # shape: [batch_size*sequence_length, hidden_size*2]
                hidden_state_ = tf.reshape(hidden_state, shape=[-1, hidden_size * 2])
                # hidden_state_: [batch_size*sequence_length, hidden_size*2]
                # W_w_attention_sentence: [hidden_size*2, hidden_size*2]
                hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_, self.W_w_attention_word)
                                                   + self.W_b_attention_word)
                # shape: [batch_size, sequence_length, hidden_size*2]
                hidden_representation = tf.reshape(hidden_representation, shape=[-1, sequence_length, hidden_size*2])

                # 1) get logits for each word in the sentence.
                # hidden_representation: [batch_size, sequence_length, hidden_size*2]
                # context_vecotor_word: [hidden_size*2]
                hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_word)
                # 对应相乘再求和，得到权重
                # shape: [batch_size, sequence_length]
                attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
                # subtract max for numerical stability (softmax is shift invariant).
                # tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
                # shape: [batch_size, 1]
                attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
                # 2) get possibility distribution for each word in the sentence.
                # shape: [batch_size, sequence_length]
                # 归一化
                p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
                # 3) get weighted hidden state by attention vector
                # shape: [batch_size, sequence_length, 1]
                p_attention_expanded = tf.expand_dims(p_attention, axis=2)
                # below sentence_representation
                # shape:[batch_size, sequence_length, hidden_size*2]<----
                # p_attention_expanded: [batch_size, sequence_length, 1]
                # hidden_state_: [batch_size, sequence_length, hidden_size*2]
                # shape: [batch_size, sequence_length, hidden_size*2]
                sentence_representation = tf.multiply(p_attention_expanded, hidden_state)
                # shape: [batch_size, hidden_size*2]
                sentence_representation = tf_utils.Mask(sentence_representation, seq_len, config.max_sent_len)
                sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
                # shape: [batch_size, hidden_size*2]
                return sentence_representation

        def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_rate=None, return_sequence=True):
            """
            Update 2017.11.21
            fix a bug
            ref: https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
            ======
            BiLSTM Layer
            Args:
                input_x: [batch, sent_len, emb_size]
                input_x_len: [batch, ]
                hidden_size: int
                num_layers: int
                dropout_rate: float
                return_sequence: True/False
            Returns:
                if return_sequence=True:
                    outputs: [batch, sent_len, hidden_size*2]
                else:
                    output: [batch, hidden_size*2]
            """

            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_size)

            # cell = tf.contrib.rnn.GRUCell(hidden_size)
            # cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            # cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size)

            if num_layers >= 1:
                # Warning! Please consider that whether the cell to stack are the same
                cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

            if dropout_rate is not None:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_rate)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_rate)

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                                  sequence_length=input_x_len, dtype=tf.float32)
            if return_sequence:
                # b_outputs: [[b, sl, h],[b, sl, h]]
                outputs = tf.concat(b_outputs, axis=2)
            else:
                # b_states: (([b, c], [b, h]), ([b, c], [b, h]))
                outputs = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
            return outputs

        def CNN(input_x, seq_len, filter_sizes, num_filters, embed_size, dropout_rate=None):
            """
            CNN Layer
            Args:
                input_x: [batch, sent_len, emb_size, 1]
                seq_len: int
                filter_sizes: list
                num_filters: int
                dropout_rate: float
            Returns:
                outputs: [batch, num_filters * len(filter_sizes)]
            """
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convolution-pooling-%s" % filter_size):
                    # ====>a.create filter
                    filter = tf.get_variable("filter-%s" % filter_size, [filter_size, embed_size, 1, num_filters],
                                             initializer=self.initializer)
                    # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                    # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                    # Conv.Returns: A `Tensor`. Has the same type as `input`.
                    # A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                    # 1) each filter with conv2d's output a shape:[1, sent_len-filter_size+1, 1, 1];2) * num_filters--->[1, sent_len - filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                    # input data format: NHWC: [batch, height, width, channels]; output:4-D

                    # shape:[batch_size, sent_len - filter_size + 1, 1, num_filters]
                    conv = tf.nn.conv2d(input_x, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    # ====>c. apply nolinearity
                    b = tf.get_variable("b-%s" % filter_size, [num_filters])
                    # shape: [batch_size, sent_len - filter_size + 1, 1, num_filters]. tf.nn.bias_add:adds `bias` to `value`
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                    # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                    # ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                    # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                    # shape:[batch_size, 1, 1, num_filters].max_pool: performs the max pooling on the input.
                    pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)
            # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
            # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
            #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
            #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
            # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
            h_pool = tf.concat(pooled_outputs, -1)
            num_filters_total = num_filters * len(filter_sizes)
            # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])
            # 4.=====>add dropout: use tf.nn.dropout
            if dropout_rate is not None:
                # [None, num_filters_total]
                outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)

            # 5. logits(use linear layer)and predictions(argmax)
            # with tf.name_scope("output"):
            #     # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            #     logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
            return outputs

        # Build the Computation Graph
        # [batch_size, sent_len, word_emd_size]
        embedded_x = tf.nn.embedding_lookup(self.we, self.input_x)
        batch_size = tf.shape(embedded_x)[0]
        if self.with_char:
            if self.char_type == 'lstm':
                # [batch_size, sent_len, word_len, char_emd_size]
                embedded_x_char = tf.nn.embedding_lookup(self.char_we, self.input_x_char)
                # batch_size = tf.shape(embedded_x_char)[0]
                # [batch_size * sent_len, word_len, char_emd_size]
                embedded_x_char = tf.reshape(embedded_x_char, [-1, self.word_len, self.char_embed_size])
                input_x_char_lens = tf.reshape(self.input_x_char_len, [-1])
                with tf.variable_scope("char_bilstm") as clstm:
                    # [batch_size * sent_len, word_len, char_emd_size]
                    char_lstm_x = BiLSTM(embedded_x_char, input_x_char_lens, self.char_lstm_size, dropout_rate=1.0, return_sequence=True)
                    char_lstm_x = char_lstm_x[:, -1, :]
                    char_x = tf.reshape(char_lstm_x, [batch_size, self.seq_len, self.char_lstm_size*2])
            if self.char_type == 'cnn':
                embedded_x_char = tf.nn.embedding_lookup(self.char_we, self.input_x_char)
                embedded_x_char = tf.reshape(embedded_x_char, [-1, self.word_len, self.char_embed_size])
                with tf.variable_scope("char_cnn") as ccnn:
                    inputs_char_embeddings_expanded = tf.expand_dims(embedded_x_char, -1)
                    char_cnn_x = CNN(inputs_char_embeddings_expanded, self.word_len, self.filter_sizes, self.num_filters,
                            self.char_embed_size, self.drop_keep_rate)
                    num_filters_total = self.num_filters * len(self.filter_sizes)
                    char_x = tf.reshape(char_cnn_x, [batch_size, self.seq_len, num_filters_total])
        if self.with_ner:
            embedded_x_ner = tf.nn.embedding_lookup(self.ner_we, self.input_x_ner)
        if self.with_pos:
            embedded_x_pos = tf.nn.embedding_lookup(self.pos_we, self.input_x_pos)
        with tf.variable_scope("seq_bilstm") as s:
            if self.with_ner:
                embedded_x = tf.concat([embedded_x, embedded_x_ner], axis=-1)
            if self.with_pos:
                embedded_x = tf.concat([embedded_x, embedded_x_pos], axis=-1)
            if self.with_char:
                embedded_x = tf.concat([embedded_x, char_x], axis=-1)
            lstm_x = BiLSTM(embedded_x, self.input_x_len, self.lstm_size, self.layer_size, self.drop_keep_rate,
                            return_sequence=True)
        if self.with_cnn:
            inputs_embeddings_expanded = tf.expand_dims(embedded_x, -1)
            cnn_x = CNN(inputs_embeddings_expanded, self.seq_len, self.filter_sizes, self.num_filters,
                        self.word_embed_size, self.drop_keep_rate)
        if self.with_cnn_lstm:
            inputs_hidden_expanded = tf.expand_dims(lstm_x, -1)
            cnn_x = CNN(inputs_hidden_expanded, self.seq_len, self.filter_sizes, self.num_filters,
                        self.lstm_size*2, self.drop_keep_rate)
        avg_pooling = tf_utils.AvgPooling(lstm_x, self.input_x_len, self.seq_len)
        max_pooling = tf_utils.MaxPooling(lstm_x, self.input_x_len)
        last_lstm = lstm_x[:, -1, :]
        last_lstm = tf.reshape(last_lstm, [batch_size, self.lstm_size * 2])
        seq_distribution = tf.concat([avg_pooling, max_pooling, last_lstm], axis=-1)
        if self.with_attention:
            attention = attention_word_level(lstm_x, self.lstm_size, self.seq_len, self.input_x_len)
            seq_distribution = tf.concat([last_lstm, attention], axis=-1)
        if self.with_rf:
            seq_distribution = tf.concat([seq_distribution, self.input_rf], axis=-1)
        if self.with_pun:
            seq_distribution = tf.concat([seq_distribution, self.input_x_pun], axis=-1)
        if self.with_senti:
            seq_distribution = tf.concat([seq_distribution, self.input_x_senti], axis=-1)
        if self.with_cnn:
            seq_distribution = tf.concat([seq_distribution, cnn_x], axis=-1)
        if self.with_cnn_lstm:
            seq_distribution = tf.concat([seq_distribution, cnn_x], axis=-1)

        hidden1 = tf.nn.relu(tf_utils.linear(seq_distribution, self.mlp_h1_size, bias=True, scope='h1'))
        logits = tf_utils.linear(hidden1, self.num_class, bias=True, scope='softmax')

        # Obtain the Predict, Loss, Train_op
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        loss = tf.reduce_mean(loss)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
        reg_loss = loss + FLAGS.lambda_l2 * l2_loss

        # Build the loss
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        # optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        if FLAGS.clipper:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.clipper)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.seq_res = hidden1
        self.logits = logits
        self.loss = loss
        self.reg_loss = reg_loss
        self.train_op = train_op
        self.global_step = global_step

    def train_model(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.input_y: batch.label,
            self.drop_keep_rate: self.FLAGS.drop_keep_rate,
            self.learning_rate: 1e-3
        }
        if self.with_char:
            feed_dict[self.input_x_char] = batch.char
            feed_dict[self.input_x_char_len] = batch.char_len
        if self.with_ner:
            feed_dict[self.input_x_ner] = batch.ner
        if self.with_pos:
            feed_dict[self.input_x_pos] = batch.pos
        if self.with_rf:
            feed_dict[self.input_rf] = batch.rf
        if self.with_pun:
            feed_dict[self.input_x_pun] = batch.pun
        if self.with_senti:
            feed_dict[self.input_x_senti] = batch.senti
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def make_feature(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.input_y: batch.label,
            self.drop_keep_rate: 1.0,
            self.learning_rate: 1e-3
        }
        if self.with_char:
            feed_dict[self.input_x_char] = batch.char
            feed_dict[self.input_x_char_len] = batch.char_len
        if self.with_ner:
            feed_dict[self.input_x_ner] = batch.ner
        if self.with_pos:
            feed_dict[self.input_x_pos] = batch.pos
        if self.with_rf:
            feed_dict[self.input_rf] = batch.rf
        if self.with_pun:
            feed_dict[self.input_x_pun] = batch.pun
        if self.with_senti:
            feed_dict[self.input_x_senti] = batch.senti
        to_return = {
            'seq_res': self.seq_res,
            'logits': self.logits,
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)

    def test_model(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.drop_keep_rate: 1.0,
        }
        if self.with_char:
            feed_dict[self.input_x_char] = batch.char
            feed_dict[self.input_x_char_len] = batch.char_len
        if self.with_ner:
            feed_dict[self.input_x_ner] = batch.ner
        if self.with_pos:
            feed_dict[self.input_x_pos] = batch.pos
        if self.with_rf:
            feed_dict[self.input_rf] = batch.rf
        if self.with_pun:
            feed_dict[self.input_x_pun] = batch.pun
        if self.with_senti:
            feed_dict[self.input_x_senti] = batch.senti
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)
