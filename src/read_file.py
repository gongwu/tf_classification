# -*- coding:utf-8 -*-
import pickle
import config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='../dic/oov.p',
                    help='input batch size for training (default: 128)')
args = parser.parse_args()

def readfile(filename):
    with open(filename) as f:
        data = pickle.load(f)
        for i in data:
            print(i)
        print(len(data))

if __name__ == '__main__':
    readfile(args.filename)