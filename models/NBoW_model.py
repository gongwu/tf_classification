# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from base.base_model import BaseModel
from utils import tf_utils


class NBoWModel(BaseModel):
    def __init__(self, config, data):
        super(NBoWModel, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Build the Computation Graph
        inputs = tf.nn.embedding_lookup(self.data.embed, self.input_x)  # [batch_size, sent_len, emd_size]
        avg_pooling = tf_utils.AvgPooling(inputs, self.input_x_len, self.seq_len)
        logits = tf_utils.linear(avg_pooling, self.num_class, bias=True, scope='softmax')

        # Obtain the Predict, Loss, Train_op
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            loss = tf.reduce_mean(loss)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
            reg_loss = loss + self.config.lambda_l2 * l2_loss
            # Build the loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            if self.config.clipper:
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.clipper)
                train_step = optimizer.apply_gradients(list(zip(grads, tvars)))
            else:
                train_step = optimizer.minimize(loss, global_step=self.global_step_tensor)
            self.predict_prob = predict_prob
            self.predict_label = predict_label
            self.logits = logits
            self.loss = loss
            self.reg_loss = reg_loss
            self.train_step = train_step
