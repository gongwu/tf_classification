# -*- coding:utf-8 _*-
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import utils
import config
import evaluation
from data import Task
from models.NBoW import NBoWModel
from models.LSTM import LSTMModel
from models.CNN import CNNModel

FLAGS = tf.flags.FLAGS
tf.set_random_seed(1234)

# File Parameters
tf.flags.DEFINE_string('log_file', 'tf-classification.log', 'path of the log file')
# tf.flags.DEFINE_string('summary_dir', 'summary', 'path of summary_dir')
# tf.flags.DEFINE_string('description', __file__, 'commit_message')
tf.flags.DEFINE_string('embed', 'SWM', 'word_embedding')
# Task Parameters
tf.flags.DEFINE_string('model', 'lstm', 'given the model name')
tf.flags.DEFINE_integer('max_epoch', 30, 'max epoches')
tf.flags.DEFINE_integer('display_step', 100, 'display each step')
tf.flags.DEFINE_integer('layer_size', 1, 'layer size')
tf.flags.DEFINE_integer('num_filters', 128, 'num_filters')
tf.flags.DEFINE_integer('threshold', 5, 'threshold')
# Hyper Parameters
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_float('drop_keep_rate', 0.9, 'dropout_keep_rate')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_float('lambda_l2', 1e-6, 'lambda_l2')
tf.flags.DEFINE_float('clipper', 30, 'clipper')
tf.flags.DEFINE_bool('init', False, 'build_vocab')
tf.flags.DEFINE_bool('isTrain', True, 'training')
tf.flags.DEFINE_bool('with_char', False, 'char_embedding')
tf.flags.DEFINE_string('char_type', 'cnn', 'char type')
tf.flags.DEFINE_bool('with_ner', False, 'ner_embedding')
tf.flags.DEFINE_bool('with_pos', False, 'pos_embedding')
tf.flags.DEFINE_bool('with_rf', False, 'rf_features')
tf.flags.DEFINE_bool('with_pun', False, 'punction_features')
tf.flags.DEFINE_bool('with_senti', False, 'sentilexi')
tf.flags.DEFINE_bool('with_attention', False, 'word_attention')
tf.flags.DEFINE_bool('with_cnn', False, 'cnn_features')
tf.flags.DEFINE_bool('with_cnn_lstm', False, 'cnn_features_hidden')
tf.flags.DEFINE_string("ckpt_dir",config.dev_model_file, "checkpoint location for the model")
FLAGS._parse_flags()


# Logger Part
logger = utils.get_logger(FLAGS.log_file)
logger.info(FLAGS.__flags)

def main():
    task = Task(init=FLAGS.init, FLAGS=FLAGS)
    FLAGS.we = task.embed
    FLAGS.char_we = task.char_embed
    FLAGS.ner_we = task.ner_embed
    FLAGS.pos_we = task.pos_embed
    FLAGS.num_vocab = task.train_data.num_vocab
    graph1 = tf.Graph().as_default()
    graph2 = tf.Graph().as_default()
    graph3 = tf.Graph().as_default()
    graph4 = tf.Graph().as_default()
    global sess_nbow
    global sess_cnn
    global sess_lstm
    global sess_rcnn
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    # with graph1: # nbow
    #     FLAGS1 = FLAGS
    #     FLAGS1.ckpt_dir = config.dev_model_file1
    #     sess_nbow = tf.Session(config=gpu_config)
    #     model_nbow = NBoWModel(FLAGS1)
    #     saver_nbow = tf.train.Saver()
    #     if os.path.exists(FLAGS1.ckpt_dir + "checkpoint"):
    #         print("Restoring Variables from Checkpoint of nbow.")
    #         saver_nbow.restore(sess_nbow, tf.train.latest_checkpoint(FLAGS1.ckpt_dir))
    #     else:
    #         print("Can't find the checkpoint.going to stop.nbow")
    #         return
    # with graph2: # cnn
    #     FLAGS2 = FLAGS
    #     FLAGS2.ckpt_dir = config.dev_model_file2
    #     sess_cnn= tf.Session(config=gpu_config)
    #     model_cnn = CNNModel(FLAGS2)
    #     saver_cnn = tf.train.Saver()
    #     if os.path.exists(FLAGS2.ckpt_dir + "checkpoint"):
    #         print("Restoring Variables from Checkpoint of cnn.")
    #         saver_cnn.restore(sess_cnn, tf.train.latest_checkpoint(FLAGS2.ckpt_dir))
    #     else:
    #         print("Can't find the checkpoint.going to stop.cnn")
    #         return
    # with graph3: # lstm
    #     FLAGS3 = FLAGS
    #     FLAGS3.ckpt_dir = config.dev_model_file3
    #     sess_lstm = tf.Session(config=gpu_config)
    #     model_lstm = LSTMModel(FLAGS3)
    #     saver_lstm = tf.train.Saver()
    #     if os.path.exists(FLAGS3.ckpt_dir + "checkpoint"):
    #         print("Restoring Variables from Checkpoint of lstm.")
    #         saver_lstm.restore(sess_lstm, tf.train.latest_checkpoint(FLAGS3.ckpt_dir))
    #     else:
    #         print("Can't find the checkpoint.going to stop.lstm")
    #         return
    with graph1: # rcnn
        FLAGS1 = FLAGS
        FLAGS1.ckpt_dir = config.dev_model_file1
        FLAGS1.with_char = True
        FLAGS1.with_attention = True
        FLAGS1.with_pos = True
        FLAGS1.with_ner = True
        FLAGS1.with_senti = True
        FLAGS1.with_pun = True
        FLAGS1.with_cnn_lstm = True
        FLAGS1.learning_rate = 0.05
        FLAGS1.char_type = 'lstm'
        sess_nbow = tf.Session(config=gpu_config)
        model_nbow = LSTMModel(FLAGS1)
        saver_nbow = tf.train.Saver()
        if os.path.exists(FLAGS1.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint of rcnn.")
            saver_nbow.restore(sess_nbow, tf.train.latest_checkpoint(FLAGS1.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop.rcnn")
            return
    with graph2: # rcnn
        FLAGS2 = FLAGS
        FLAGS2.ckpt_dir = config.dev_model_file2
        FLAGS2.with_char = True
        FLAGS2.with_attention = True
        FLAGS2.with_pos = True
        FLAGS2.with_ner = True
        FLAGS2.with_senti = True
        FLAGS2.with_pun = True
        FLAGS2.with_cnn_lstm = True
        FLAGS2.drop_keep_rate = 0.7
        FLAGS2.char_type = 'lstm'
        sess_cnn = tf.Session(config=gpu_config)
        model_cnn = LSTMModel(FLAGS2)
        saver_cnn = tf.train.Saver()
        if os.path.exists(FLAGS2.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint of rcnn.")
            saver_cnn.restore(sess_cnn, tf.train.latest_checkpoint(FLAGS2.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop.rcnn")
            return
    with graph3: # rcnn
        FLAGS3 = FLAGS
        FLAGS3.ckpt_dir = config.dev_model_file3
        FLAGS3.with_char = True
        FLAGS3.with_attention = True
        FLAGS3.with_pos = True
        FLAGS3.with_ner = True
        FLAGS3.with_senti = True
        FLAGS3.with_pun = True
        FLAGS3.with_cnn_lstm = True
        FLAGS3.drop_keep_rate = 0.5
        FLAGS3.char_type = 'lstm'
        sess_lstm = tf.Session(config=gpu_config)
        model_lstm = LSTMModel(FLAGS3)
        saver_lstm = tf.train.Saver()
        if os.path.exists(FLAGS3.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint of rcnn.")
            saver_lstm.restore(sess_lstm, tf.train.latest_checkpoint(FLAGS3.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop.rcnn")
            return

    with graph4: # rcnn
        FLAGS4 = FLAGS
        FLAGS4.ckpt_dir = config.dev_model_file4
        FLAGS4.with_char = True
        FLAGS4.with_attention = True
        FLAGS4.with_pos = True
        FLAGS4.with_ner = True
        FLAGS4.with_senti = True
        FLAGS4.with_pun = True
        FLAGS4.with_cnn_lstm = True
        sess_rcnn = tf.Session(config=gpu_config)
        model_rcnn = LSTMModel(FLAGS4)
        saver_rcnn = tf.train.Saver()
        if os.path.exists(FLAGS4.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint of rcnn.")
            saver_rcnn.restore(sess_rcnn, tf.train.latest_checkpoint(FLAGS4.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop.rcnn")
            return
    with tf.Session(config=gpu_config) as sess:
        def get_logits(dataset):
            batch_size = 100  # for Simple
            probs = []
            for batch in dataset.batch_iter(batch_size):
                nbow_prob = np.array(model_nbow.test_model(sess_nbow, batch)['predict_prob'])
                cnn_prob = np.array(model_cnn.test_model(sess_cnn, batch)['predict_prob'])
                lstm_prob= np.array(model_lstm.test_model(sess_lstm, batch)['predict_prob'])
                rcnn_prob = np.array(model_rcnn.test_model(sess_rcnn, batch)['predict_prob'])
                sum_prob = 0.25 * nbow_prob + 0.25 * cnn_prob + 0.25 * lstm_prob + 0.25 * rcnn_prob
                probs.append(sum_prob)
            probs = np.concatenate(probs, 0)
            labels = tf.cast(tf.argmax(probs, 1), tf.int32)
            return probs, labels

        probs, labels = get_logits(task.dev_data)
        with open(config.dev_predict_ensemble_file, 'w') as f:
            for label in labels.eval():
                f.write(str(int(label))+"\n")


if __name__ == '__main__':
    main()
