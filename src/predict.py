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
            return macro_f1, preds, golds

        macro_f1, dev_result, golds = do_eval(task.dev_data)
        test_macro_f1, test_result, golds_test = do_eval(task.test_data)
        logger.info(
            'dev = {:.5f}, test = {:.5f}'.format(macro_f1, test_macro_f1),
            # 'dev = {:.5f}'.format(macro_f1)
        )
        with open(config.dev_predict_file, 'w') as f:
            for label in dev_result:
                f.write(str(int(label))+"\n")
        with open(config.test_predict_file, 'w') as f:
            for label in test_result:
                f.write(str(int(label))+"\n")
        # with open(config.dev_gold_final_file, 'w') as f:
        #     for label in golds:
        #         f.write(str(int(label))+'\n')

class Predict(object):

    def test_model(self, file_list, result_file_path):
        if type(file_list) != list:
            file_list = [file_list]
        preds = []
        for file in file_list:
            pred = []
            with codecs.open(file, 'r', encoding='utf8') as f:
                for line in f:
                    label = int(line.strip())
                    pred.append(label)
            preds.append(pred)
        # Adaboost
        # predicts = map(lambda x: int(float(x.strip())), open(config.RESULT_Adaboost_FILE_PATH))
        # preds.append(predicts)

        final_pred = []
        weight = 1
        for item in zip(*preds):
            # for pred1, pred2, pred3 in zip(*preds):
            pred1 = item[0]
            pred2 = item[1]
            pred3 = item[2]
            vote_result = self.match(int(pred1), int(pred2), int(pred3), weight)
            # avg = sum(item) / float(len(item))
            final_pred.append(vote_result)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, final_pred)))

        return final_pred

        # 三个数匹配，大多数投票。若三个数各不相同，则取第weight个。weight:1,2,3

    def match(self, num1, num2, num3, weight):
        if num1 == num2 == num3:  # 三者相同
            return num1
        elif num1 != num2 != num3:  # 三者互不相同
            # if num2 == -1:
            #     return num2
            # else:
            if weight == 1:
                return num1
            elif weight == 2:
                return num2
            else:
                return num3
        else:  # 两者相同
            if num1 == num2 or num1 == num3:
                return num1
            else:
                return num2

if __name__ == '__main__':
    # file_list = [config.test_predict_all_file, config.test_predict_file, config.test_predict_nlp]
    # predict = Predict()
    # predict.test_model(file_list, config.test_predict_ensemble_file_final)
    main()