# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import re
import itertools
import evaluation
import config
import json

# def f(p=[]):
#     p.append(1)
#     return p
#
# def g(p=[]):
#     p.append(1)
#     return p
#
# a = f()
# print(a)
# b = g()
# print(b)
# c = f()
# print(a, b, c)
# a = tf.constant(np.array([[1, 2, 3]]))
# b = tf.constant(np.array([[4, 5, 6]]))
'''python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
  tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
  '''
# c = tf.concat(0, [a, b])
# # c = [a, b]
# with tf.Session() as sess:
#     print(sess.run(c))
# def lstm():
#     return [1]
# print([lstm() for _ in range(2)])

# for word in re.split('\\s+', "hello world"):
#     for char in word:
#         print(char)

# chars = set()
# chars.update("hello")
# chars.update("world")
# char_lens = []
# examples = [[["hello", "world"], ["life", "is", "short"]], [["hello", "world"], ["life", "is", "short"]]]
# for i in range(len(examples)):
#     char_lens.append([min(len(word), 4) for word in examples[i]])
# print(char_lens)
# char_lens = np.array(char_lens, dtype=np.int32)
# print(char_lens)

# x = [[[1, 2, 3],
#       [4, 5, 6]],
#      [[7, 8, 9],
#       [10, 11, 12]]]
# with tf.Session() as sess:
#     print(sess.run(tf.transpose(x, perm=[1, 2, 0])))
#     print(sess.run(tf.transpose(x, perm=[1, 0, 2])))
#     print(sess.run(tf.transpose(x, perm=[2, 1, 0])))
#     print(sess.run(tf.transpose(x, perm=[2, 0, 1])))
# a = [[[1, 2, 3], [1, 2, 3]]]
# b = [[[1, 2, 3], [4, 5, 6]]]
# c = [[[1, 2, 3], [7, 8, 9]]]
# a = tf.constant(a)
# b = tf.constant(b)
# c = tf.constant(c)
# with tf.Session() as sess:
#     d = tf.concat([a, b, c], axis=-1)
#     print(sess.run(d))
# print(list(itertools.chain(*list)))
# a = [[1, 2, 3], [4, 5, 6]]
# print(list(itertools.chain(*a)))
# tokens = [[
# "RuPaul",
# "'s",
# "Drag",
# "Race",
# "bingo",
# "fun",
# "."
#  ],[
# "Drag",
# "Queens",
# "be",
# "SEXY",
# "!"
#  ],[
# "#rupaulsdragrace",
# "@user",
# "abwyman",
# "#la",
# "..."
# ]]
# ners = [[
# "ORGANIZATION",
# "O",
# "O",
# "O",
# "O",
# "O",
# "O"
# ],
# [
# "O",
# "O",
# "O",
# "O",
# "O",
# "O",
# "O"
# ],
# [
# "O",
# "O",
# "O",
# "O",
# "O"
#  ]]

# wanted_tokens = data_utils._process_ngram_tokens(tokens, ners)
# print(list(itertools.chain(*wanted_tokens)))
# word = "ilovemyjob"
# print(data_utils.hashtagSegment(word))
# print(list(set(a).intersection(set(b))))

# examples = data.read_data(config.train_file)
# np.random.shuffle(examples)
# print(examples[0])
# print(examples[1])
# print(examples[2])
# counts = np.zeros(6)
# for example in examples:
#     sents = example[0]
#     if len(sents) <= 10:
#         counts[0] += 1
#     elif len(sents) > 10 and len(sents) <= 20:
#         counts[1] += 1
#     elif len(sents) > 20 and len(sents) <= 30:
#         counts[2] += 1
#     elif len(sents) > 30 and len(sents) <= 40:
#         counts[3] += 1
#     elif len(sents) > 40 and len(sents) <= 50:
#         counts[4] += 1
#     else: counts[5] += 1
# for count in counts:
#     print(count)
# examples_train = examples[:int(0.9 * len(examples))]
# print(examples_train[0][0])
# examples_dev = examples[int(0.9 * len(examples)):]
# print(examples_dev[0][0])
# data_utils.cout_distribution(examples_dev)
# examples_test = data.read_data(config.dev_new_file)
# print(examples_test[0][0])
# a = np.array([[1, 2, 3],[4, 5, 6]])
# b = np.array([[7, 8, 9], [10, 11, 12]])
# c = 0.5 * a + b
# print(c)
# list = json.load(open(config.dev_file_final, "r"), encoding='utf-8')
# json.dump(list, open(config.dev_file_final_new, 'w'), indent=2, ensure_ascii=False)

with open(config.test_predict_ensemble_file_final) as f1, open(config.test_predict_nlp) as f2:
    gold1 = []
    gold2 = []
    for line1 in f1:
        gold1.append(int(line1))
    for line2 in f2:
        gold2.append(int(line2))
    print(len(gold1))
    print(len(gold2))
    n = 0
    for i in range(len(gold1)):
        if gold1[i] != gold2[i]:
            n+=1
    print(n)
