# coding: utf-8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils


class NBoWModel(object):

    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS
        self.config = config
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.seq_len = config.max_sent_len
        self.embed_size = config.word_dim
        self.num_class = config.num_class
        self.mlp_h1_size = 140
        self.mlp_h2_size = 140

        # Add PlaceHolder
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sent_len]
        self.input_x_len = tf.placeholder(tf.int32, (None,))
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))

        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.drop_hidden1 = tf.placeholder(tf.float32)
        self.drop_hidden2 = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Add Word Embedding
        self.we = tf.Variable(FLAGS.we, name='emb')

        # Build the Computation Graph
        inputs = tf.nn.embedding_lookup(self.we, self.input_x)  # [batch_size, sent_len, emd_size]
        avg_pooling = tf_utils.AvgPooling(inputs, self.input_x_len, self.seq_len)
        hidden1 = tf.nn.relu(tf_utils.linear(avg_pooling, self.mlp_h1_size, bias=True, scope='h1'))
        hidden1_drop = tf.nn.dropout(hidden1, keep_prob=self.drop_hidden1)
        hidden2 = tf.nn.relu(tf_utils.linear(hidden1_drop, self.mlp_h2_size, bias=True, scope='h2'))
        hidden2_drop = tf.nn.dropout(hidden2, keep_prob=self.drop_hidden2)
        logits = tf_utils.linear(hidden2_drop, self.num_class, bias=True, scope='softmax')
        # logits = tf_utils.linear(avg_pooling, self.num_class, bias=True, scope='softmax')

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
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.seq_res = hidden2_drop
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
            self.drop_hidden1: 1.0,
            self.drop_hidden2: 1.0,
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
            self.drop_keep_rate: 1.0,
            self.drop_hidden1: 1.0,
            self.drop_hidden2: 1.0,
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
            self.drop_hidden1: 1.0,
            self.drop_hidden2: 1.0,
        }
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)
