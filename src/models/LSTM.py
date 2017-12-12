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
        self.seq_len = config.max_sent_len
        self.word_len = config.max_word_len
        self.word_embed_size = config.word_dim
        self.char_embed_size = config.char_dim
        self.num_class = config.num_class
        self.char_lstm_size = 50
        self.lstm_size = 512
        self.layer_size = FLAGS.layer_size
        self.with_char = FLAGS.with_char
        self.with_ner = FLAGS.with_ner
        self.with_pos = FLAGS.with_pos
        self.with_attention = FLAGS.with_attention
        self.drop_keep_rate = tf.placeholder(tf.float32)
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
        if self.with_char:
            self.input_x_char = tf.placeholder(tf.int32, (None, self.seq_len,
                                                          self.word_len))  # [batch_size, sent_len, word_len]
            self.input_x_char_len = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sen_len]
            # The Char Embedding is Random Initialization
            self.char_we = tf.Variable(FLAGS.char_we, name='char_emb')


        # attention process:
        # 1.get logits for each word in the sentence.
        # 2.get possibility distribution for each word in the sentence.
        # 3.get weighted sum for the sentence as sentence representation.
        def attention_word_level(hidden_state, hidden_size, sequence_length, scope=None, reuse=None):
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

        # Build the Computation Graph
        # [batch_size, sent_len, word_emd_size]
        embedded_x = tf.nn.embedding_lookup(self.we, self.input_x)

        if self.with_char:
            # [batch_size, sent_len, word_len, char_emd_size]
            embedded_x_char = tf.nn.embedding_lookup(self.char_we,
                                                     self.input_x_char)
            batch_size = tf.shape(embedded_x_char)[0]
            # [batch_size * sent_len, word_len, char_emd_size]
            embedded_x_char = tf.reshape(embedded_x_char, [-1, self.word_len,
                                                           self.char_embed_size])
            input_x_char_lens = tf.reshape(self.input_x_char_len, [-1])
            with tf.variable_scope("char_bilstm") as c:
                # [batch_size * sent_len, word_len, char_emd_size]
                char_x = BiLSTM(embedded_x_char, input_x_char_lens, self.char_lstm_size, dropout_rate=1.0, return_sequence=True)
                char_x = char_x[:, -1, :]
                char_x = tf.reshape(char_x, [batch_size, self.seq_len, self.char_lstm_size*2])
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
        avg_pooling = tf_utils.AvgPooling(lstm_x, self.input_x_len, self.seq_len)
        max_pooling = tf_utils.MaxPooling(lstm_x, self.input_x_len)
        if self.with_attention:
            attention = attention_word_level(lstm_x, self.lstm_size, self.seq_len)
            seq_distribution = attention
        else:
            seq_distribution = tf.concat([avg_pooling, max_pooling], axis=-1)
        logits = tf_utils.linear(seq_distribution, self.num_class, bias=True, scope='softmax')

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
        # optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        if FLAGS.clipper:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.clipper)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
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
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def test_model(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.drop_keep_rate: 1.0
        }
        if self.with_char:
            feed_dict[self.input_x_char] = batch.char
            feed_dict[self.input_x_char_len] = batch.char_len
        if self.with_ner:
            feed_dict[self.input_x_ner] = batch.ner
        if self.with_pos:
            feed_dict[self.input_x_pos] = batch.pos
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)