import tensorflow as tf
import numpy as np
import re
import itertools
import evaluation

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
