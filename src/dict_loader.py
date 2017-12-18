# -*- coding:utf-8 _*-
from utils import singleton
import numpy as np
import config


@singleton
class Dict_loader(object):
    def __init__(self):
        # Sentiment_Lexicon
        self.dict_BL = self._dict_Senti_Lexi_0(config.LEXI_BL)
        self.dict_GI = self._dict_Senti_Lexi_0(config.LEXI_GI)
        self.dict_IMDB = self._dict_Senti_Lexi_0(config.LEXI_IMDB)
        self.dict_MPQA = self._dict_Senti_Lexi_0(config.LEXI_MPQA)
        self.dict_NRCE = self._dict_Senti_Lexi_0(config.LEXI_NRCEMOTION)
        self.dict_AF = self._dict_Senti_Lexi_1(config.LEXI_AFINN)
        self.dict_NRC140_U = self._dict_Senti_Lexi_1(config.LEXI_NRC140_U)
        self.dict_NRCH_U = self._dict_Senti_Lexi_1(config.LEXI_NRCHASHTAG_U)
        self.dict_NRC140_B = self._dict_Senti_Lexi_2(config.LEXI_NRC140_B)
        self.dict_NRCH_B = self._dict_Senti_Lexi_2(config.LEXI_NRCHASHTAG_B)


    def _load_dict_cashtag_to_mean_score(self, in_path):
        dict_cashtag_to_mean_score = {}
        with open(in_path) as fin:
            for line in fin:
                cash_tag, score = line.strip().split("\t")
                score = float(score)
                dict_cashtag_to_mean_score[cash_tag] = score

        return dict_cashtag_to_mean_score

# in_file: vocab 对应的vector文件
    def _load_dict_word_to_vec(self, in_file):
        dict_word_to_vec = {}
        with open(in_file) as fin:
            for line in fin:
                word, vec_string = line.split(" ", 1)
                vec = np.fromstring(vec_string, dtype='float32', sep=' ')
                dict_word_to_vec[word] = vec
        return dict_word_to_vec

    def _dict_Senti_Lexi_0(slef, fLexi):
        """Bing Liu & General Inquirer & imdb & MPQA & NRCEmotion"""
        #format: word \t positive_score \t negative_score
        dict_ = {}

        f = open(fLexi)
        for line in f:
            line = line.strip().split("\t")
            score = float(line[1]) - float(line[-1])
            dict_[line[0]] = score

        return dict_

    def _dict_Senti_Lexi_1(slef, fLexi):
        """AFINN & NRC140_U & NRCHash_U"""
        #format: word \t score
        dict_ = {}

        for line in open(fLexi):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[line[0]] = score

        return dict_

    def _dict_Senti_Lexi_2(slef, fLexi):
        """NRC140_B & NRCHash_B"""
        dict_ = {}

        for line in open(fLexi):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[tuple(line[0].split(" "))] = score

        return dict_

    def _dict_word_cluster(self, in_file):
        dict_word_cluster = {}
        with open(in_file) as fin:
            for line in fin:
                word, label = line.split(" ")
                dict_word_cluster[word] = int(label)
            return dict_word_cluster


