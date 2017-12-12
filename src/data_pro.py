# -*- coding:utf-8 -*-
import pickle
import codecs
import config
import data
import numpy as np


with codecs.open(config.n2i_file, 'rb') as f:
    data = pickle.load(f)
print(type(data))
# examples = data.read_data(config.train_file)
# dev_examples = examples[int(0.9*len(examples)):]
# with open(config.train_data_file, 'wb') as f:
#     for example in examples:
#         for word in example[0]:
#             f.write(word+' ')
#         f.write('\n')

# label_count = np.zeros(20)
#
# with open(config.dev_gold_file, 'r') as f:
#     for line in f.readlines():
#         label_count[int(line.strip())] += 1
# sum = np.sum(label_count)
# for i in label_count:
#     print("%d \t %.2f%%" % (i, (i / sum * 100)))