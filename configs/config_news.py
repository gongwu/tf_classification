import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    # Root dir
    ROOT = '../experiments/news'
    DATA_DIR = '../data/nlpcc_data'
    VOCABULARY_DIR = '../vocabulary'
    # dir path
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join(ROOT, config.exp_name, "summary")
    config.checkpoint_dir = os.path.join(ROOT, config.exp_name, "checkpoint/")
    config.dic_dir = os.path.join(DATA_DIR, "dic")
    config.result_dir = os.path.join(ROOT, config.exp_name, "output")
    # file path
    config.train_file = os.path.join(DATA_DIR, "word", "train.txt")
    config.dev_file = os.path.join(DATA_DIR, "word", "dev.txt")
    config.test_file = None
    config.dev_predict_file = os.path.join(config.result_dir, "dev_predict_file.txt")
    config.word_embed_file = os.path.join(DATA_DIR, "embed/emb_wd", "embedding.300")
    config.w2i_file = os.path.join(config.dic_dir, "w2i.p")
    config.c2i_file = os.path.join(config.dic_dir, "c2i.p")
    config.n2i_file = os.path.join(config.dic_dir, "n2i.p")
    config.p2i_file = os.path.join(config.dic_dir, "p2i.p")
    config.oov_file = os.path.join(config.dic_dir, "oov.p")
    config.we_file = os.path.join(config.dic_dir, "we.p")
    config.VOCAB_NORMAL_WORDS_PATH = os.path.join(VOCABULARY_DIR, "normal_word.pkl")
    # param
    config.max_sent_len = 29
    config.max_word_len = 26
    config.num_class = 20
    config.word_dim = 300
    config.char_dim = 50
    config.ner_dim = 50
    config.pos_dim = 50
    # label
    config.category2id = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4, 'world': 5, 'finance': 6, 'game': 7,
                            'travel': 8, 'military': 9, 'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
                             'story': 15, 'regimen': 16, 'essay': 17}
    config.id2category = {index: label for label, index in config.category2id.items()}
    return config
