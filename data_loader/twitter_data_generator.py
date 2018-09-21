import numpy as np

from base.data_generator import DataGenerator
from utils import data_utils
from utils import utils


class TwitterDataGenerator(DataGenerator):
    def __init__(self, config):
        super(TwitterDataGenerator, self).__init__(config)
        self.max_sent_len = self.config.max_sent_len
        self.max_word_len = self.config.max_word_len
        self.train_file = self.config.train_file
        self.dev_file = self.config.dev_file
        self.test_file = self.config.test_file
        self.segmenter = utils.Segmenter(self.config.VOCAB_NORMAL_WORDS_PATH)
        self.init_embedding()

    def build_data(self, data_file):
        """
                return the formatted matrix, which is used as the input to deep learning models
                Args: file_list:
                      word_vocab:
                """
        self.examples = data_utils.read_json_data(data_file, self.segmenter)
        y = []
        sent_features = []
        sent_lens = []
        ids = []
        for example in self.examples:
            sents = example[0]
            label = int(example[1])
            ners = example[2]
            poses = example[3]
            lemmas = example[4]
            ids.append(example[5])
            char = data_utils.char_to_matrix(lemmas, self.char_vocab)
            sent = data_utils.sent_to_index(sents, self.word_vocab)
            ner = data_utils.ner_to_index(ners, self.ner_vocab)
            pos = data_utils.pos_to_index(poses, self.pos_vocab)
            one_hot_label = data_utils.onehot_vectorize(label, self.config.num_class)
            y.append(one_hot_label)
            # 有的句子长度为0, 取平均长度
            if len(sent) == 0:
                sent = np.ones(8)
            sent_features.append((sent, char, ner, pos))
            sent_lens.append(min(len(sent), self.max_sent_len))
        # 这里添加char, ner的特征， 之后再做处理
        f_sents = []
        f_chars = []
        f_ners = []
        f_poses = []
        char_lens = []
        for feature in sent_features:
            f_sents.append(feature[0])
            f_chars.append(feature[1])
            f_ners.append(feature[2])
            f_poses.append(feature[3])
        input_x = data_utils.pad_2d_matrix(f_sents, self.max_sent_len)
        input_x_ner = data_utils.pad_2d_matrix(f_ners, self.max_sent_len)
        input_x_pos = data_utils.pad_2d_matrix(f_poses, self.max_sent_len)
        input_x_char = data_utils.pad_3d_tensor(f_chars, self.max_sent_len, self.max_word_len)

        for i in range(len(input_x_char)):
            char_lens.append([min(len(word), self.max_word_len) for word in input_x_char[i]])
        x_len = sent_lens
        x_char_len = char_lens
        self.input_x = np.array(input_x, dtype=np.int32)  # [batch_size, sent_len]
        self.input_x_ner = np.array(input_x_ner, dtype=np.int32)
        self.input_x_pos = np.array(input_x_pos, dtype=np.int32)
        self.input_x_char = np.array(input_x_char, dtype=np.int32)
        self.x_len = np.array(x_len, dtype=np.int32)  # [batch_size]
        self.x_char_len = np.array(x_char_len, dtype=np.int32)
        self.y = np.array(y, dtype=np.float32)  # [batch_size, class_number]
        self.ids = np.array(ids, dtype=np.float32)

    def next_batch(self, batch_size, shuffle=False):
        input_x = self.input_x
        input_x_char = self.input_x_char
        input_x_ner = self.input_x_ner
        input_x_pos = self.input_x_pos
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
            batch.add('sent_len', x_len[excerpt])
            batch.add('char_len', x_char_len[excerpt])
            batch.add('label', y[excerpt])
            batch.add('ids', ids[excerpt])
            yield batch

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

        examples = data_utils.read_json_data(file_list, self.segmenter)
        sents = []
        ners = []
        poses = []
        lemmas = []
        for example in examples:
            sent = example[0]
            ner = example[2]
            pos = example[3]
            lemma = example[4]
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

    def init_embedding(self):
        self.word_embed_file = self.config.word_embed_file
        self.word_dim = self.config.word_dim
        self.char_dim = self.config.char_dim
        self.ner_dim = self.config.ner_dim
        self.pos_dim = self.config.pos_dim
        self.threshold = self.config.threshold

        self.we_file = self.config.we_file
        self.w2i_file = self.config.w2i_file
        self.c2i_file = self.config.c2i_file
        self.n2i_file = self.config.n2i_file
        self.p2i_file = self.config.p2i_file

        # the char_embed always init
        if self.config.init:
            self.word_vocab, self.char_vocab, self.ner_vocab, self.pos_vocab = self.build_vocab()
            self.embed = data_utils.load_word_embedding(self.word_vocab, self.word_embed_file, self.config, self.word_dim)
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
        self.char_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.char_vocab), self.char_dim)),
                                   dtype=np.float32)
        self.ner_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.ner_vocab), self.ner_dim)), dtype=np.float32)
        self.pos_embed = np.array(np.random.uniform(-0.25, 0.25, (len(self.pos_vocab), self.pos_dim)), dtype=np.float32)
        print("vocab size: %d" % len(self.word_vocab), "we shape: ", self.embed.shape)

