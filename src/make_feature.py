# -*- coding:utf-8 _*-
from __future__ import print_function

import codecs
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
tf.flags.DEFINE_integer('threshold', 1, 'threshold')
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
    if FLAGS.model == 'nbow':
        model = NBoWModel(FLAGS)
    elif FLAGS.model == 'lstm':
        model = LSTMModel(FLAGS)
    elif FLAGS.model == 'cnn':
        model = CNNModel(FLAGS)
    else:
        raise NotImplementedError

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        saver = tf.train.Saver()
        print("Restoring Variables from Checkpoint")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

        def make_feature(dataset):
            batch_size = 100  # for Simple
            golds, features, ids = [], [], []
            for batch in dataset.batch_iter(batch_size):
                results = model.make_feature(sess, batch)
                golds.append(np.argmax(batch.label, 1))
                features.append(results['seq_res'])
                ids.append(batch.ids)
            golds = np.concatenate(golds, 0)
            features = np.concatenate(features, 0)
            ids = np.concatenate(ids, 0)
            return golds, features, ids

        train_golds, train_features, train_ids = make_feature(task.train_data)
        dev_golds, dev_features, dev_ids = make_feature(task.dev_data)
        test_golds, test_features, test_ids = make_feature(task.test_data)
        print(len(test_features[0]))
        with open(config.train_feature_file, 'w') as f:
            assert len(train_golds) == len(train_features)
            for i in range(len(train_features)):
                f.write(str(train_golds[i]) + ' ')
                for j in range(len(train_features[i])):
                    if int(train_features[i][j]) == 0:
                        continue
                    f.write(str(j + 1) + ':' + str(train_features[i][j])+' ')
                f.write('# '+str(train_ids[i])+'\n')
        with open(config.dev_feature_file, 'w') as f:
            assert len(dev_golds) == len(dev_features)
            for i in range(len(dev_features)):
                f.write(str(dev_golds[i]) + ' ')
                for j in range(len(dev_features[i])):
                    if int(dev_features[i][j]) == 0:
                        continue
                    f.write(str(j + 1) + ':' + str(dev_features[i][j]) + ' ')
                f.write('# '+str(dev_ids[i]) + '\n')
        with open(config.test_feature_file, 'w') as f:
            assert len(test_golds) == len(test_features)
            for i in range(len(test_features)):
                f.write(str(test_golds[i]) + ' ')
                for j in range(len(test_features[i])):
                    if int(test_features[i][j]) == 0:
                        continue
                    f.write(str(j + 1) + ':' + str(test_features[i][j]) + ' ')
                f.write('# '+str(test_ids[i]) + '\n')

if __name__ == '__main__':
    main()
