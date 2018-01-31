# coding: utf8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils
import data_utils


class CNNModel(object):

    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS
        self.config = config
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.seq_len = config.max_sent_len
        self.embed_size = config.word_dim
        self.num_class = config.num_class
        self.filter_sizes = [1, 2, 3, 4]
        self.num_filters = FLAGS.num_filters
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        # Add PlaceHolder
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sent_len]
        self.input_x_len = tf.placeholder(tf.int32, (None,))
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))
        # self.mlp_h1_size = 200
        # self.mlp_h2_size = 140
        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.drop_hidden1 = tf.placeholder(tf.float32)
        self.drop_hidden2 = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Add Word Embedding
        self.we = tf.Variable(FLAGS.we, name='emb')

        # Build the Computation Graph
        def CNN(input_x, seq_len, filter_sizes, num_filters=1, dropout_rate=None):
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
                    filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, num_filters],
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
                    b = tf.get_variable("b-%s" % filter_size, [num_filters])  # ADD 2017-06-09
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

        # TODO: implenment CNN Begin:
        inputs = tf.nn.embedding_lookup(self.we, self.input_x)  # [batch_size, sent_len, emd_size]
        inputs_embeddings_expanded = tf.expand_dims(inputs, -1)
        cnn_x = CNN(inputs_embeddings_expanded, self.seq_len, self.filter_sizes, self.num_filters, self.drop_keep_rate)
        # TODO: implenment CNN end
        # hidden1 = tf.nn.relu(tf_utils.linear(cnn_x, self.mlp_h1_size, bias=True, scope='h1'))
        # hidden1_drop = tf.nn.dropout(hidden1, keep_prob=self.drop_keep_rate)
        # hidden2 = tf.nn.relu(tf_utils.linear(hidden1_drop, self.mlp_h2_size, bias=True, scope='h2'))
        # hidden2_drop = tf.nn.dropout(hidden2, keep_prob=self.drop_keep_rate)
        logits = tf_utils.linear(cnn_x, self.num_class, bias=True, scope='softmax')

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
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.seq_res = cnn_x
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
            # self.drop_hidden1: 0.8,
            # self.drop_hidden2: 0.8,
            self.learning_rate: 1e-3
        }
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
            # self.drop_hidden1: 1.0,
            # self.drop_hidden2: 1.0,
            self.learning_rate: 1e-3
        }
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
            # self.drop_hidden1: 1.0,
            # self.drop_hidden2: 1.0,
        }
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)

if __name__ == '__main__':
    model = CNNModel()
    input_x = data_utils.pad_2d_matrix([1, 3, 5], 30)
    with tf.Session() as sess:
        model.train_model(sess, [[]])
