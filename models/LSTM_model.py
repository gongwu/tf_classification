# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from base.base_model import BaseModel
from utils import tf_utils


class LSTMModel(BaseModel):
    def __init__(self, config, data):
        super(LSTMModel, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Build the Computation Graph
        self.layers = self.config.layers
        self.lstm_size = self.config.lstm_size
        inputs = tf.nn.embedding_lookup(self.data.embed, self.input_x)  # [batch_size, sent_len, emd_size]

        def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_keep_rate=None, return_sequence=True):
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_size)

            def gru_cell():
                return tf.contrib.rnn.GRUCell(hidden_size)

            cell_fw = lstm_cell()
            cell_bw = lstm_cell()

            if num_layers > 1:
                cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

            if dropout_keep_rate is not None:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_rate)

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                                  sequence_length=input_x_len, dtype=tf.float32)
            if return_sequence:
                outputs = tf.concat(b_outputs, axis=2)
            else:
                # states: [c, h]
                outputs = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
            return outputs

        with tf.variable_scope("bilstm") as s:
            lstm_x = BiLSTM(inputs, self.input_x_len, self.lstm_size,
                            num_layers=self.layers,
                            dropout_keep_rate=self.drop_keep_rate,
                            return_sequence=True)

        avg_pooling = tf_utils.AvgPooling(inputs, self.input_x_len, self.seq_len)
        max_pooling = tf_utils.MaxPooling(lstm_x, self.input_x_len)
        logits = tf_utils.linear([max_pooling, avg_pooling], self.num_class, bias=True, scope='softmax')

        # Obtain the Predict, Loss, Train_op
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        loss = tf.reduce_mean(loss)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
        reg_loss = loss + self.config.lambda_l2 * l2_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.config.clipper:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.clipper)
            train_step = optimizer.apply_gradients(list(zip(grads, tvars)))
        else:
            train_step = optimizer.minimize(loss, global_step=self.global_step_tensor)
        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.loss = loss
        self.reg_loss = reg_loss
        self.train_step = train_step