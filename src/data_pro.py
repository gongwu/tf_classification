# -*- coding:utf-8 -*-
import pickle
import codecs
import config
import numpy as np
import itertools
from collections import Counter
from text_unidecode import unidecode
import json

train_list = []
dev_list = []
dev_label = []
test_list = []
notoverlap_list = []
notoverlap_label = []
with open(config.train_es_file, 'r') as f1, open(config.dev_es_file, 'r') as f2:
    for line in f1.readlines():
        train_list.append(line.split("\t#\t")[1].strip())
    for line in f2.readlines():
        dev_list.append(line.split("\t#\t")[1].strip())
        dev_label.append(line.split("\t#\t")[0].strip())
    # for line in f3.readlines():
    #     test_list.append(line.split("\t")[0].strip())
    # for line in f4.readlines():
    #     notoverlap_list.append(line.split("\t")[1].strip())
    #     notoverlap_label.append(line.split("\t")[0].strip())

# train_list = ["Guess what flavor this is I just approved it and it's coming soon. #NothinButaPeanut..."]
# dev_list = ["Guess what flavor this is I just approved it and it's coming soon. #NothinButaPeanut..."]
# print(test_list[0][-1])
# a = set(train_list)
# b = set(dev_list)
train_dev_overlap_list = list(set(train_list).intersection(set(dev_list)))
print(len(train_dev_overlap_list))
print(len(dev_list))
print(len(train_dev_overlap_list)/float(len(dev_list)))
# train_test_overlap_list = list(set(train_list).intersection(set(test_list)))
# notoverlap = []
# for i in range(50000):
#     if dev_list[i] not in train_dev_overlap_list:
#         notoverlap.append(i)
# print(len(train_dev_overlap_list))
# print(len(notoverlap))
# print(len(train_dev_overlap_list)+len(notoverlap))
# print(len(dev_list))
# with open(config.dev_new_file, 'w') as f:
#     for i in overlap:
#         f.write(dev_label[i])
#         f.write("\t")
#         f.write(dev_list[i])
#         f.write("\n")

# list = []
# for i in range(len(notoverlap_list)):
#     info = {}
#     info['text'] = notoverlap_list[i]
#     info["label"] = notoverlap_label[i]
#     list.append(info)
# with open(config.TRIAL_DATA_JSON_NEW, "w") as f:
#     json.dump(list, f, indent=2, ensure_ascii=False)
#     print("加载入文件完成...")
