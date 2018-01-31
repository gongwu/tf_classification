# coding: utf-8
from __future__ import print_function
import data_utils
import config
import numpy as np
import codecs
np.random.seed(1234)


def read_es_data(file_list):
    """
    load data from file list
    Args: file_list:
    Returns:
    """
    if type(file_list) != list:
        file_list = [file_list]

    examples = []
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                label = items[0]
                sent = items[1].split()
                examples.append((sent, label))
    return examples


def read_data(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    examples = []
    for file in file_list:
        tweets = data_utils.load_tweets(file)
        for tweet in tweets:
            # add the feature
            sents = data_utils.get_text_unigram(tweet)
            lemmas = data_utils.get_text_lemmas(tweet)
            ners = data_utils.get_text_ner(tweet)
            pos = data_utils.get_text_pos(tweet)
            punction = data_utils.get_text_punction(tweet)
            senti = data_utils.sentilexi(tweet)
            label = tweet["label"]
            id = 0
            # if file == config.train_file:
            #     id = tweet["id"]
            text = tweet["cleaned_text"]
            examples.append((sents, label, ners, pos, punction, senti, lemmas, id, text))
    return examples


class Dataset(object):
    def __init__(self, examples,
                 type,
                 word_vocab,
                 char_vocab,
                 ner_vocab,
                 pos_vocab,
                 rf_vocab,
                 max_sent_len,
                 max_word_len,
                 num_class):
        """
        return the formatted matrix, which is used as the input to deep learning models
        Args: file_list:
              word_vocab:
        """
        self.num_vocab = len(word_vocab)
        # examples = read_data(file_list)
        # np.random.shuffle(examples)
        # if type == 'train':
        #     self.examples = examples[:int(0.9*len(examples))]
        # elif type == 'dev':
        #     print("dev")
        #     self.examples = examples[int(0.9*len(examples)):]
        #     data_utils.cout_distribution(self.examples)
        # else:
        self.examples = examples
        # examples = self.examples
        y = []
        ids = []
        for example in examples:
            label = int(example[1])
            one_hot_label = data_utils.onehot_vectorize(label, num_class)
            y.append(one_hot_label)
            ids.append(example[7])
        #
        sent_features = []
        sent_lens = []
        for example in examples:
            sents = example[0]
            ners = example[2]
            poses = example[3]
            pun = example[4]
            senti = example[5]
            lemmas = example[6]
            char = data_utils.char_to_matrix(lemmas, char_vocab)
            sent = data_utils.sent_to_index(sents, word_vocab)
            ner = data_utils.ner_to_index(ners, ner_vocab)
            pos = data_utils.pos_to_index(poses, pos_vocab)
            rf = data_utils.rf_to_dict(sents, rf_vocab, word_vocab)

            # 有的句子长度为0, 取平均长度
            if len(sent) == 0:
                sent = np.ones(8)
            sent_features.append((sent, char, ner, pos, rf, pun, senti))
            sent_lens.append(min(len(sent), max_sent_len))
        # 这里添加char, ner的特征， 之后再做处理
        f_sents = []
        f_chars = []
        f_ners = []
        f_poses = []
        f_rf = []
        f_pun = []
        f_senti = []
        char_lens = []
        for feature in sent_features:
            f_sents.append(feature[0])
            f_chars.append(feature[1])
            f_ners.append(feature[2])
            f_poses.append(feature[3])
            f_rf.append(feature[4])
            f_pun.append(feature[5])
            f_senti.append(feature[6])
        input_x = data_utils.pad_2d_matrix(f_sents, max_sent_len)
        input_x_ner = data_utils.pad_2d_matrix(f_ners, max_sent_len)
        input_x_pos = data_utils.pad_2d_matrix(f_poses, max_sent_len)
        input_x_char = data_utils.pad_3d_tensor(f_chars, max_sent_len, max_word_len)

        for i in range(len(input_x_char)):
            char_lens.append([min(len(word), max_word_len) for word in input_x_char[i]])
        x_len = sent_lens
        x_char_len = char_lens
        self.input_x = np.array(input_x, dtype=np.int32)  # [batch_size, sent_len]
        self.input_x_ner = np.array(input_x_ner, dtype=np.int32)
        self.input_x_pos = np.array(input_x_pos, dtype=np.int32)
        self.input_x_char = np.array(input_x_char, dtype=np.int32)
        self.input_x_rf = np.array(f_rf)
        self.input_x_pun = np.array(f_pun, dtype=np.int32)
        self.input_x_senti = np.array(f_senti, dtype=np.float32)
        self.x_len = np.array(x_len, dtype=np.int32)  # [batch_size]
        self.x_char_len = np.array(x_char_len, dtype=np.int32)
        self.y = np.array(y, dtype=np.float32)  # [batch_size, class_number]
        self.ids = np.array(ids, dtype=np.float32)

    def batch_iter(self, batch_size, shuffle=False):
        """
        UPDATE_0: add Batch for yield
        To support different model with different data:
        - model_1 want data 1, 2, 3, 4;
        - model_2 want data 1, 2, 3, 4, 5;
        ===
        during training: add some data to be enough batch_size
        during test: add some data to be enough batch_size
        :param batch_size:
        :param shuffle:
        :return:
        """
        input_x = self.input_x
        input_x_char = self.input_x_char
        input_x_ner = self.input_x_ner
        input_x_pos = self.input_x_pos
        input_x_pun = self.input_x_pun
        input_x_senti = self.input_x_senti
        input_x_rf = self.input_x_rf
        x_len = self.x_len
        x_char_len = self.x_char_len
        y = self.y
        ids = self.ids
        assert len(input_x) == len(y)
        n_data = len(y)

        idx = np.arange(n_data)
        if shuffle:
            idx = np.random.permutation(n_data)

        for start_idx in range(0, n_data, batch_size):
            # end_idx = min(start_idx + batch_size, n_data)
            end_idx = start_idx + batch_size
            excerpt = idx[start_idx:end_idx]

            batch = data_utils.Batch()
            batch.add('sent', input_x[excerpt])
            batch.add('char', input_x_char[excerpt])
            batch.add('ner', input_x_ner[excerpt])
            batch.add('pos', input_x_pos[excerpt])
            batch.add('rf', data_utils.fill_2d_matrix(input_x_rf[excerpt], self.num_vocab))
            batch.add('pun', input_x_pun[excerpt])
            batch.add('senti', input_x_senti[excerpt])
            batch.add('sent_len', x_len[excerpt])
            batch.add('char_len', x_char_len[excerpt])
            batch.add('label', y[excerpt])
            batch.add('ids', ids[excerpt])
            yield batch


class Task(object):

    def __init__(self, init=False, FLAGS=None):
        self.FLAGS = FLAGS
        self.train_file = config.train_file
        self.dev_file = config.dev_new_file
        # 测试集之后添加
        self.test_file = config.test_file_final
        if FLAGS.embed == "SWM":
            self.word_embed_file = config.word_embed_SWM
        elif FLAGS.embed == "google":
            self.word_embed_file = config.word_embed_google
        elif FLAGS.embed == 'w2v':
            self.word_embed_file = config.word_embed_w2v
        elif FLAGS.embed == 'glove':
            self.word_embed_file = config.word_embed_glove
        self.word_dim = config.word_dim
        self.char_dim = config.char_dim
        self.ner_dim = config.ner_dim
        self.pos_dim = config.pos_dim
        self.max_sent_len = config.max_sent_len
        self.max_word_len = config.max_word_len
        self.num_class = config.num_class
        self.threshold = FLAGS.threshold

        self.we_file = config.we_file
        self.w2i_file = config.w2i_file
        self.c2i_file = config.c2i_file
        self.n2i_file = config.n2i_file
        self.p2i_file = config.p2i_file
        self.rf2i_file = config.rf_file
        self.train_predict_file = None
        self.dev_predict_file = None
        self.test_predict_file = None

        # the char_embed always init
        if init:
            self.word_vocab, self.char_vocab, self.ner_vocab, self.pos_vocab = self.build_vocab()
            self.embed = data_utils.load_word_embedding(self.word_vocab, self.word_embed_file, self.word_dim)
            data_utils.save_params(self.word_vocab, self.w2i_file)
            data_utils.save_params(self.char_vocab, self.c2i_file)
            data_utils.save_params(self.ner_vocab, self.n2i_file)
            data_utils.save_params(self.pos_vocab, self.p2i_file)
            data_utils.save_params(self.embed, self.we_file)
        else:
            self.embed = data_utils.load_params(self.we_file)
            self.word_vocab = data_utils.load_params(self.w2i_file)
            self.char_vocab = data_utils.load_params(self.c2i_file)
            self.ner_vocab = data_utils.load_params(self.n2i_file)
            self.pos_vocab = data_utils.load_params(self.p2i_file)
            self.embed = self.embed.astype(np.float32)
        self.rf_vocab = data_utils.load_key_value_dict_from_file(self.rf2i_file)
        self.char_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.char_vocab), self.char_dim)), dtype=np.float32)
        self.ner_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.ner_vocab), self.ner_dim)), dtype=np.float32)
        self.pos_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.pos_vocab), self.pos_dim)), dtype=np.float32)
        print("vocab size: %d" % len(self.word_vocab), "we shape: ", self.embed.shape)
        # examples = read_data(self.train_file)
        # np.random.shuffle(examples)
        # examples_train = examples[:int(0.9*len(examples))]
        # print(examples_train[0][0])
        # examples_dev = examples[int(0.9*len(examples)):]
        # print(examples_dev[0][0])
        examples_train = read_data(self.train_file)
        examples_dev = read_data(self.dev_file)
        data_utils.cout_distribution(examples_dev)
        examples_test = read_data(self.test_file)
        self.train_data = Dataset(examples_train, 'none', self.word_vocab, self.char_vocab, self.ner_vocab, self.pos_vocab, self.rf_vocab, self.max_sent_len, self.max_word_len, self.num_class )
        self.dev_data = Dataset(examples_dev, 'dev', self.word_vocab, self.char_vocab, self.ner_vocab, self.pos_vocab, self.rf_vocab, self.max_sent_len, self.max_word_len, self.num_class)
        if self.test_file:
            self.test_data = Dataset(examples_test, 'none', self.word_vocab, self.char_vocab, self.ner_vocab, self.pos_vocab, self.rf_vocab, self.max_sent_len, self.max_word_len, self.num_class)

    def build_vocab(self):
        """
            build sents is for build vocab
            during multi-lingual task, there are two kinds of sents
            :return: sents
        """
        if self.test_file is None:
            print('test_file is None')
            file_list = [self.train_file, self.dev_file]
        else:
            file_list = [self.train_file, self.dev_file, self.test_file]

        examples = read_data(file_list)
        sents = []
        ners = []
        poses = []
        lemmas = []
        for example in examples:
            sent = example[0]
            ner = example[2]
            pos = example[3]
            lemma = example[6]
            sents.append(sent)
            ners.append(ner)
            poses.append(pos)
            lemmas.append(lemma)
        word_vocab = data_utils.build_word_vocab(sents, self.threshold)
        ner_vocab = data_utils.build_ner_vocab(ners)
        pos_vocab = data_utils.build_pos_vocab(poses)
        char_vocab = data_utils.build_char_vocab(lemmas)

        # 统计平均长度与最大长度
        max_sent_len = 0
        avg_sent_len = 0
        for sent in sents:
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            avg_sent_len += len(sent)
        avg_sent_len /= len(sents)
        print('task: max_sent_len: {}'.format(max_sent_len))
        print('task: avg_sent_len: {}'.format(avg_sent_len))
        max_word_len = 0
        avg_word_len = 0
        total_len = 0
        for sent in sents:
            for word in sent:
                word = list(word)
                if len(word) > max_word_len:
                    max_word_len = len(word)
                avg_word_len += len(word)
            total_len += len(sent)
        avg_word_len /= total_len
        print('task: max_word_len: {}'.format(max_word_len))
        print('task: avg_word_len: {}'.format(avg_word_len))
        return word_vocab, char_vocab, ner_vocab, pos_vocab


if __name__ == '__main__':
    pass