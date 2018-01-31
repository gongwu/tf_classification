# coding: utf8
import random
import numpy as np
import re, os
import pickle
import json
import itertools
import utils
import config
import nltk
from collections import Counter
from dict_loader import Dict_loader


pad_word = '__PAD__'
unk_word = '__UNK__'
Segmenter = utils.Segmenter(config.VOCAB_NORMAL_WORDS_PATH)  # 对hashtag进行分词
set_neg = set([t.strip() for t in open(config.NEGATION_PATH)])
punc = set([".", ",", "?", "!", "...", ";"])


def load_tweets(file_path):
    tweet_list = json.load(open(file_path, "r"), encoding="utf-8")
    return tweet_list


def set_dict_key_value(dict, key):
    if key not in dict:
        dict[key] = 0
    dict[key] += 1


def get_text_unigram(microblog):
    tokens = microblog["parsed_text"]["tokens"]  # clean_text做预处理得到的分词结果
    ners = microblog["parsed_text"]["ners"]
    pos = microblog["parsed_text"]["pos"]
    wanted_tokens = _process_ngram_tokens(tokens, pos, ners)  # 去掉各种number及长度小于2的词
    return list(itertools.chain(*wanted_tokens))


def get_text_lemmas(microblog):
    tokens = microblog["parsed_text"]["lemmas"]  # clean_text做预处理得到的分词结果
    ners = microblog["parsed_text"]["ners"]
    pos = microblog["parsed_text"]["pos"]
    wanted_tokens = _process_ngram_tokens(tokens, pos, ners)  # 去掉各种number及长度小于2的词
    return list(itertools.chain(*wanted_tokens))


def load_key_value_dict_from_file(dict_file_path):
    dict = {}
    dict_file = open(dict_file_path)
    lines = [line.strip() for line in dict_file]
    dict_file.close()
    for line in lines:
        if line == "":
            continue
        key, value = line.split("\t")
        dict[key] = eval(value)
    return dict


def get_text_ner(microblog):
    ners = microblog["parsed_text"]["ners"]
    return list(itertools.chain(*ners))  # 将多个list拼为1个list


def get_text_pos(microblog):
    poss = microblog["parsed_text"]["pos"]
    return list(itertools.chain(*poss))  # 将多个list拼为1个list


# 是否包含！，是否包含多个！，是否包含？，是否包含多个？，是否包含？！或！？
# 最后一个token中是否包含！，最后一个token中是否包含？，！的个数，？的个数
def get_text_punction(microblog):
    has_exclamation = 0
    has_several_exclamation = 0
    has_question = 0
    has_several_question = 0
    has_exclamation_question = 0
    end_exclamation = 0
    end_question = 0
    num_exclamation = 0
    num_question = 0
    # print microblog["parsed_text"]["tokens"]
    if microblog["parsed_text"]["tokens"]:
        tokens = []  # 本句子的所有tokens
        token_lists = microblog["parsed_text"]["tokens"]
        for token_list in token_lists:
            for word in token_list:
                tokens.append(word)
            if "!" in tokens[-1]:
                end_exclamation = 1
            if "?" in tokens[-1]:
                end_question = 1
        sentence = " ".join(tokens)
        # print tokens
        exclamation_list = re.findall("!", sentence)
        num_exclamation = len(exclamation_list)
        if len(exclamation_list) != 0:  # 无感叹号
            has_exclamation = 1
            if len(exclamation_list) > 2:
                has_several_exclamation = 1

        question_list = re.findall("\?", sentence)
        num_question = len(question_list)
        if len(question_list) != 0:
            has_question = 1
            if len(question_list) > 2:
                has_several_question = 1

        excla_ques_list = re.findall("!\?", sentence)
        ques_excla_list = re.findall("\?!", sentence)
        if excla_ques_list or ques_excla_list:
            has_exclamation_question = 1
        if "!" in tokens[-1]:
            end_exclamation = 1
        if "?" in tokens[-1]:
            end_question = 1
    feature = [has_exclamation, has_several_exclamation, has_question, has_several_question, has_exclamation_question]
    feature.append(end_exclamation)
    feature.append(end_question)
    feature.append(num_exclamation)
    feature.append(num_question)
    return feature


def sentilexi(microblog):
    feature = []
    # dict的value值都是1维score（若字典中本来有pos_score和neg_score，则pos_score-neg_score）
    Lexicon_dict_list = [
        Dict_loader().dict_BL,
        Dict_loader().dict_GI,
        Dict_loader().dict_IMDB,
        Dict_loader().dict_MPQA,
        Dict_loader().dict_NRCE,
        Dict_loader().dict_AF,
        Dict_loader().dict_NRC140_U,
        Dict_loader().dict_NRCH_U
    ]

    # tokens = list(itertools.chain(*
    # 将否定词后的4个词加上_NEG后缀
    tokens = reverse_neg(microblog)

    for Lexicon in Lexicon_dict_list:
        score = []
        for word in tokens:
            flag = -0.8 if word.endswith("_NEG") else 1
            word = word.replace("_NEG", "")
            if word in Lexicon:
                score.append(Lexicon[word] * flag)

        if len(score) == 0:
            feature += [0] * 11
            continue

        countPos, countNeg, countNeu = 0, 0, 0
        length = len(score) * 1.0
        for s in score:
            if s > 0.49:
                countPos += 1
            elif s < -0.49:
                countNeg += 1
            else:
                countNeu += 1

        feature += [countPos, countNeg, countNeu, countPos / length, countNeg / length, countNeu / length, max(score),
                    min(score)]

        finalscore = sum(score)
        # feature.append(finalscore)
        if finalscore > 0:
            feature += [1, 0]
        elif finalscore < 0:
            feature += [0, 1]
        else:
            feature += [0, 0]

        # pos_score = [t for t in score if t > 0]
        # neg_score = [t for t in score if t < 0]
        # feature.append(sum(pos_score))
        # feature.append(sum(neg_score))

        # if pos_score:
        #     feature.append(pos_score[-1])
        # else:
        #     feature.append(0)
        # if neg_score:
        #     feature.append(neg_score[-1])
        # else:
        #     feature.append(0)

        word = tokens[-1]
        flag = -0.8 if word.endswith("_NEG") else 1
        word = word.replace("_NEG", "")
        if word in Lexicon:
            feature.append(Lexicon[word] * flag)
        else:
            feature.append(0)

    # Bigram Lexicons
    for Lexicon in [Dict_loader().dict_NRC140_B, Dict_loader().dict_NRCH_B]:
        score = []
        bigram = list(nltk.ngrams(tokens, 2))
        for index, bi in enumerate(bigram):
            flag = -0.8 if bi[0].endswith("_NEG") and bi[1].endswith("_NEG") else 1
            bi = (bi[0].replace("_NEG", ""), bi[1].replace("_NEG", ""))
            bigram[index] = bi
            if bi in Lexicon:
                score.append(Lexicon[bi] * flag)
        if not score:
            feature += [0] * 11
            continue

        countPos, countNeg, countNeu = 0, 0, 0
        length = len(score) * 1.0
        for s in score:
            if s > 0.49:
                countPos += 1
            elif s < -0.49:
                countNeg += 1
            else:
                countNeu += 1

        feature += [countPos, countNeg, countNeu, countPos / length, countNeg / length, countNeu / length, max(score),
                    min(score)]

        finalscore = sum(score)
        # feature.append(finalscore)
        if finalscore > 0:
            feature += [1, 0]
        elif finalscore < 0:
            feature += [0, 1]
        else:
            feature += [0, 0]

        pos_score = [t for t in score if t > 0]
        neg_score = [t for t in score if t < 0]
        # feature.append(sum(pos_score))
        # feature.append(sum(neg_score))
        # if pos_score:
        #     feature.append(pos_score[-1])
        # else:
        #     feature.append(0)
        # if neg_score:
        #     feature.append(neg_score[-1])
        # else:
        #     feature.append(0)
        bi = bigram[-1]
        flag = -0.8 if bi[0].endswith("_NEG") and bi[1].endswith("_NEG") else 1
        bi = (bi[0].replace("_NEG", ""), bi[1].replace("_NEG", ""))
        if bi in Lexicon:
            feature.append(Lexicon[bi] * flag)
        else:
            feature.append(0)
    return feature


# 将否定词后的4个词加上_NEG后缀
def reverse_neg(microblog):
    mtoken = []
    tokens = list(itertools.chain(*microblog["parsed_text"]["tokens"]))
    sentence = " ".join(tokens)
    length = len(tokens)

    index = 0
    while(index != length):
        cur_token = tokens[index].lower()
        mtoken.append(cur_token)
        if cur_token in set_neg or cur_token.endswith("n't"):
            for i in range(index + 1, min(length, index + 4)):  # 将否定词后的4个词带上"_NEG"
                index = i
                cur_token_1 = tokens[i].lower()
                if tokens[i] in punc:  # 若遇到标点符号则停止加"_NEG"
                    mtoken.append(cur_token_1)
                    break
                mtoken.append(cur_token_1 + "_NEG")
        index += 1
    return mtoken


def removeItemsInDict(dict, threshold=1):
    if threshold > 1:
        for key in list(dict.keys()):
            if key == pad_word or key == unk_word:
                continue
            if dict[key] < threshold:
                dict.pop(key)
    return dict


def _process_ngram_tokens(tokens, pos, ners):
    wanted_tokens = []
    for sent_words, sent_pos, sent_ners in zip(tokens, pos, ners):
        wanted_sent_words = []
        for word, pos, ner in zip(sent_words, sent_pos, sent_ners):
            # 去掉各种number
            if ner in ["DATE", "NUMBER", "MONEY", "PERCENT"]:
                # word = ner
                continue
            # if utils.pos2tag(pos) == "#":
            #     continue
            # 将包含数字和单词的token替换成NUMBER_WORD
            if re.search("([0-9]*\.?[0-9]+)", word):
                # word = "NUMBER_WORD"
                continue

            # 去掉hashtag变小写
            # while word.startswith("#"):
            #     word = word[1:].lower()
            # 将hashtag去掉#，加入到句子中
            tag = 0
            while word.startswith("#"):
                word = word[1:].lower()
                tag = 1
            if tag == 1:
                if len(word) >= 2:
                    words = hashtagSegment(word)
                    wanted_sent_words += words
                    continue
                else:
                    continue
            # 去掉这些标点符号开头的token
            tag = 0
            punctuations = ["@", "'", ":", ";", "?", "!", "=", "_", "^", "*", "-", ".", "`"]
            for punctuation in punctuations:
                if word.startswith(punctuation):
                #     word = word[1:].lower()
                # elif word.endswith(punctuation):
                #     word = word[:-1].lower()
                    tag = 1
                    break
            if tag == 1:
                continue
            if word.strip() == "":
                continue
            # 去掉长度小于2的词
            if len(word) < 2:
                continue
            word = word.lower()
            wanted_sent_words.append(word)
        wanted_tokens.append(wanted_sent_words)
    return wanted_tokens


def hashtagSegment(word):
    token2 = []
    token1 = (Segmenter.get(word)).split(" ")  # 对hashtag进行分词
    for word_ in token1:
        if len(word_) >= 2:
            token2.append(word_)
    return token2


def save_params(params, fname):
    """
    Pickle uses different protocols to convert your data to a binary stream.
    - In python 2 there are 3 different protocols (0, 1, 2) and the default is 0.
    - In python 3 there are 5 different protocols (0, 1, 2, 3, 4) and the default is 3.
    You must specify in python 3 a protocol lower than 3 in order to be able to load
    the data in python 2. You can specify the protocol parameter when invoking pickle.dump.
    """
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=2)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def make_batches(size, batch_size):
    """
    make batch index according to batch_size and size
    :param size: the size of dataset
    :param batch_size: the size of batch
    :return: list: [(0, batch_size), (batch_size, 2*batch_size), ..., (. , min(., .))]
    """
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def vectorize(score, num_class):
    """
    NOT suitable for classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    score = float(score)
    ceil, floor = int(np.ceil(score)), int(np.floor(score))
    if ceil == floor:
        one_hot[floor - 1] = 1
    else:
        one_hot[floor - 1] = ceil - score
        one_hot[ceil - 1] = score - floor
    one_hot = one_hot + 0.00001
    return one_hot


def onehot_vectorize(label, num_class):
    """
    For classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    one_hot[label] = 1.0
    return one_hot


def bow_vectorize(sent_dic, num_vocab):
    bow = np.zeros(num_vocab, dtype=np.float32)
    for idx, value in sent_dic.items():
        bow[idx] = value
    return bow


def sent_to_index(sent, word_vocab):
    """
    :param sent:
    :param word_vocab:
    :return:
    """
    sent_index = []
    for word in sent:
        if word not in word_vocab:
            sent_index.append(word_vocab[unk_word])
        else:
            sent_index.append(word_vocab[word])
    return sent_index


def ner_to_index(ners, ner_vocab):
    """
    :param sent:
    :param ner_vocab:
    :return:
    """
    ner_index = []
    for ner in ners:
        if ner not in ner_vocab:
            ner_index.append(ner_vocab[unk_word])
        else:
            ner_index.append(ner_vocab[ner])
    return ner_index


def pos_to_index(poses, pos_vocab):
    """
    :param sent:
    :param pos_vocab:
    :return:
    """
    pos_index = []
    for pos in poses:
        if pos not in poses:
            pos_index.append(pos_vocab[unk_word])
        else:
            pos_index.append(pos_vocab[pos])
    return pos_index


def rf_to_dict(sents, rf_vocab, word_vocab):
    rf_tweet = {}
    for word in sents:
        if word not in rf_vocab:
            tf = 0
        else:
            tf = rf_vocab[word]
        rf_tweet[word] = tf
    new_feat_dict = {}
    for word in rf_tweet:
        if word in word_vocab:
            new_feat_dict[word_vocab[word]] = rf_tweet[word]
    return new_feat_dict


def rf_to_vector(sents, rf_vocab, word_vocab):
    rf_tweet = {}
    for word in sents:
        if word not in rf_vocab:
            tf = 0
        else:
            tf = rf_vocab[word]
        rf_tweet[word] = tf
    new_feat_dict = np.zeros(len(word_vocab), dtype=np.float32)
    for word in rf_tweet:
        if word in word_vocab:
            new_feat_dict[word_vocab[word]] = rf_tweet[word]
    return new_feat_dict


def get_feature_by_feat_dict(dict, feat_dict):
    new_feat_dict = {}
    for feat in feat_dict:
        if feat in dict:
            new_feat_dict[dict[feat]] = feat_dict[feat]
    return new_feat_dict


def char_to_matrix(sent, char_vocab):
    """
    :param sent
    :param char_vocab
    :return:
    """
    char_matrix = []
    for word in sent:
        char_index = []
        for char in word:
            if char not in char_vocab:
                char_index.append(char_vocab[unk_word])
            else:
                char_index.append(char_vocab[char])
        char_matrix.append(char_index)
    return char_matrix


def pad_1d_vector(words, max_sent_len, dtype=np.int32):
    # 大于最大长度截断， 小于最大长度补0
    padding_words = np.zeros((max_sent_len, ), dtype=dtype)
    kept_length = len(words)
    if kept_length > max_sent_len:
        kept_length = max_sent_len
    padding_words[:kept_length] = words[:kept_length]
    return padding_words


def pad_2d_matrix(batch_words, max_sent_len=None, dtype=np.int32):
    """
    :param batch_words: [batch_size, sent_length]
    :param max_sent_len: if None, max(sent_length)
    :param dtype:
    :return: padding_words: [batch_size, max_sent_length], 0
    """

    if max_sent_len is None:
        max_sent_len = np.max([len(words) for words in batch_words])

    batch_size = len(batch_words)
    padding_words = np.zeros((batch_size, max_sent_len), dtype=dtype)

    for i in range(batch_size):
        words = batch_words[i]
        kept_length = len(words)
        if kept_length > max_sent_len:
            kept_length = max_sent_len
        padding_words[i, :kept_length] = words[:kept_length]
    return padding_words


def pad_3d_tensor(batch_chars, max_sent_length=None, max_word_length=None, dtype=np.int32):
    """
    :param batch_chars: [batch_size, sent_length, word_length]
    :param max_sent_length:
    :param max_word_length:
    :param dtype:
    :return:
    """
    if max_sent_length is None:
        max_sent_length = np.max([len(words) for words in batch_chars])

    if max_word_length is None:
        max_word_length = np.max([np.max([len(chars) for chars in words]) for words in batch_chars])

    batch_size = len(batch_chars)
    padding_chars = np.zeros((batch_size, max_sent_length, max_word_length), dtype=dtype)

    for i in range(batch_size):
        sent_length = max_sent_length
        # 不按最大长度补齐
        if len(batch_chars[i]) < max_sent_length:
            sent_length = len(batch_chars[i])

        for j in range(sent_length):
            chars = batch_chars[i][j]
            kept_length = len(chars)
            if kept_length > max_word_length:
                kept_length = max_word_length
            padding_chars[i, j, :kept_length] = chars[:kept_length]
    return padding_chars


def fill_2d_matrix(batch_f_rf, num_vocab, dtype=np.float32):
    batch_size = len(batch_f_rf)
    padding_rf = np.zeros((batch_size, num_vocab), dtype=dtype)
    for i in range(batch_size):
        for idx, value in batch_f_rf[i].items():
            padding_rf[i][idx] = value
    return padding_rf


def build_word_vocab(sents, threshold=1):
    """
    :param sents:
    :return: word2index
    """
    dictionary = {}
    for sent in sents:
        for word in sent:
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1
    print(len(sents))
    print(len(dictionary))
    dictionary = removeItemsInDict(dictionary, threshold)
    print(len(dictionary))
    words_vocab = {str(key): index + 2 for index, key in enumerate(sorted(dictionary.keys()))}
    # words_vocab = {word: index+2 for index, word in enumerate(words)}
    words_vocab[pad_word] = 0
    words_vocab[unk_word] = 1
    # words_vocab = removeItemsInDict(words_vocab, threshold)
    return words_vocab


def build_ner_vocab(ners):
    """
   :param ners:
   :return: ner2index
   """
    ner_set = set()
    for ner in ners:
        ner_set.update(ner)
    ners_vocab = {ner: index + 2 for index, ner in enumerate(ner_set)}
    return ners_vocab


def build_pos_vocab(poses):
    """
   :param ners:
   :return: ner2index
   """
    pos_set = set()
    for pos in poses:
        pos_set.update(pos)
    pos_vocab = {pos: index + 2 for index, pos in enumerate(pos_set)}
    return pos_vocab


def build_char_vocab(sents):
    """
    :param sents:
    :return: char2index
    """
    chars = set()
    for sent in sents:
        for word in sent:
            word = list(word)
            chars.update(word)
    chars_vocab = {char: index+2 for index, char in enumerate(chars)}
    chars_vocab[pad_word] = 0
    chars_vocab[unk_word] = 1
    return chars_vocab


def load_fasttext_unk_words(oov_word_list, word2index, word_embedding):
    pass


def load_fasttext(word2index, emb_file, n_dim=100):
    """
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param word2index: dict, word2index['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :return: np.array(n_words, n_dim)
    """
    pass


def load_word_embedding(word2index, emb_file, n_dim=300):
    """
    UPDATE_1: fix the
    ===
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param word2index: dict, word2index['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :return: np.array(n_words, n_dim)
    """
    print('Load word embedding: %s' % emb_file)

    assert word2index[pad_word] == 0
    assert word2index[unk_word] == 1

    pre_trained = {}
    n_words = len(word2index)

    embeddings = np.random.uniform(-0.25, 0.25, (n_words, n_dim))
    embeddings[0, ] = np.zeros(n_dim)

    with open(emb_file, 'r') as f:
    # with open(emb_file, 'r', errors='ignore') as f:
        for idx, line in enumerate(f):
            # 第一行可能是维度和行数
            if idx == 0 and len(line.split()) == 2:
                continue
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])
            # 词
            w = ''.join(sp[0:len(sp) - n_dim])
            # 词向量
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            if w in word2index and w not in pre_trained:
                embeddings[word2index[w]] = emb
                pre_trained[w] = 1

    pre_trained_len = len(pre_trained)

    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))
    # 未登录词
    oov_word_list = [w for w in word2index if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open(config.oov_file, 'wb'))

    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


def load_embed_from_text(emb_file, token_dim):
    """
    :return: embed: numpy, vocab2id: dict
    """
    print('==> loading embed from txt')

    embed = []
    vocab2id = {}

    word_id = 0
    embed.append([0.0] * token_dim)

    with open(emb_file, 'r') as fr:

        print('embedding info: ', fr.readline())

        for line in fr:
            t = line.rstrip().split()
            word_id += 1
            vocab2id[t[0]] = word_id

            # python3 map return a generator not a list
            embed.append(list(map(float, t[1:])))

    print('==> finished load input embed from txt')
    return np.array(embed, dtype=np.float32), vocab2id


class Batch(object):
    """
    Tricks:
    1. setattr and getattr
    2. __dict__ and vars
    """
    def __init__(self):

        pass

    def add(self, name, value):
        setattr(self, name, value)

    def get(self, name):
        if name == 'self':
            value = self.__dict__  # or value = vars(self)
        else:
            value = getattr(self, name)
        return value


def read_pure_data(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    examples = []
    for file in file_list:
        tweets = load_tweets(file)
        for tweet in tweets:
            sents = get_text_unigram(tweet)
            label = tweet["label"]
            examples.append((label, sents))
    return examples


def cout_distribution(examples):
    label_count = np.zeros(20)
    for example in examples:
        label_count[int(example[1])] += 1
    sum = np.sum(label_count)
    for i in label_count:
        print(i / sum * 100), "%", " ", i


def create_top_key():
    examples = read_pure_data(config.train_file)
    data_dict = {}
    for example in examples:
        if int(example[0]) not in data_dict:
            data_dict[int(example[0])] = example[1]
        else:
            data_dict[int(example[0])].extend(example[1])

    stop_word = []
    with open(config.STOP_WORD_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            stop_word.append(line)

    with open(config.top_key_file, 'w') as f:
        for key in data_dict.keys():
            data_dict[key] = sorted(dict(Counter(data_dict[key])).items(), key=lambda d: d[1], reverse=True)
            top_key = []
            for value in data_dict[key]:
                if value[0] not in stop_word:
                    top_key.append(value)
                    if len(top_key) == 200:
                        break
            f.write(str(key) + "\t")
            for word in top_key:
                f.write(word[0] + '\t')
            f.write('\n')
