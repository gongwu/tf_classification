# -*- coding:utf-8 -*-
import pickle
import codecs
import config


with codecs.open(config.w2i_file, 'rb') as f:
    data = pickle.load(f)
for i in data:
    print(i)