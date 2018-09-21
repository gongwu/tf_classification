# -*- coding:utf-8 -*-
from base.base_train import BaseTrain


class LSTMTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(LSTMTrainer, self).__init__(sess, model, data, config, logger)

    def train_step(self, batch):
        feed_dict = {
            self.model.input_x: batch.sent,
            self.model.input_x_len: batch.sent_len,
            self.model.input_y: batch.label,
            self.model.drop_keep_rate: self.config.drop_keep_rate,
            self.model.learning_rate: self.config.learning_rate
        }
        to_return = {
            'train_step': self.model.train_step,
            'loss': self.model.loss,
        }
        return self.sess.run(to_return, feed_dict)

    def test_step(self, batch):
        feed_dict = {
            self.model.input_x: batch.sent,
            self.model.input_x_len: batch.sent_len,
            self.model.drop_keep_rate: 1.0,
        }
        to_return = {
            'predict_label': self.model.predict_label,
            'predict_prob': self.model.predict_prob
        }
        return self.sess.run(to_return, feed_dict)
