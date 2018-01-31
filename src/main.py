# coding: utf-8
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
tf.flags.DEFINE_string("ckpt_dir", config.dev_model_file, "checkpoint location for the model")
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

    best_macro_f1 = 0.
    best_dev_result = []
    # best_test_macro_f1 = 0.
    best_test_result = []
    with tf.Session(config=gpu_config) as sess:
        saver = tf.train.Saver(max_to_keep=1)
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        total_batch = 0
        early_stop = 0
        curr_epoch = sess.run(model.epoch_step)
        for epoch in range(curr_epoch, FLAGS.max_epoch):
            for batch in task.train_data.batch_iter(FLAGS.batch_size, shuffle=True):
                total_batch += 1
                results = model.train_model(sess, batch)
                step = results['global_step']
                loss = results['loss']
                if total_batch % FLAGS.display_step == 0:
                    print('batch_{} steps_{} cost_val: {:.5f}'.format(total_batch, step, loss))
                    logger.info('==>  Epoch {:02d}/{:02d}: '.format(epoch, total_batch))

            def do_eval(dataset):
                batch_size = 100  # for Simple
                preds, golds = [], []
                for batch in dataset.batch_iter(batch_size):
                    results = model.test_model(sess, batch)
                    preds.append(results['predict_label'])
                    golds.append(np.argmax(batch.label, 1))
                preds = np.concatenate(preds, 0)
                golds = np.concatenate(golds, 0)
                predict_labels = [config.id2category[predict] for predict in preds]
                gold_labels = [config.id2category[gold] for gold in golds]
                overall_accuracy, macro_p, macro_r, macro_f1 = evaluation.Evaluation_all(predict_labels, gold_labels)
                return macro_f1, preds

            macro_f1, dev_result = do_eval(task.dev_data)
            test_macro_f1, test_result = do_eval(task.test_data)
            logger.info(
                'dev = {:.5f}, test = {:.5f}'.format(macro_f1, test_macro_f1),
                # 'dev = {:.5f}'.format(macro_f1)
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_dev_result = dev_result
                best_test_result = test_result
                saver.save(sess, FLAGS.ckpt_dir+'model.ckpt', epoch)
                print("Model saved.")
                logger.info(
                    'dev = {:.5f} best!!!!, test = {:.5f}'.format(best_macro_f1, test_macro_f1)
                    # 'dev = {:.5f} best!!!!'.format(best_macro_f1)
                )
                early_stop = 0
            else:
                early_stop += 1
            # if test_macro_f1 > best_test_macro_f1:
            #     best_test_macro_f1 = test_macro_f1
            #     best_test_result = test_result
            #     logger.info(
            #         'dev = {:.5f}, test = {:.5f} best!!!!'.format(macro_f1, best_test_macro_f1)
            #         # 'dev = {:.5f} best!!!!'.format(best_macro_f1)
            #     )
            if early_stop >= 5:
                break
            sess.run(model.epoch_increment)
        with open(config.dev_predict_file, 'w') as f:
            for label in best_dev_result:
                f.write(str(int(label))+"\n")
        with open(config.test_predict_file, 'w') as f:
            for label in best_test_result:
                f.write(str(int(label))+"\n")


if __name__ == '__main__':
    main()
