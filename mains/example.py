import tensorflow as tf
from configs.config_twitter import process_config
# from configs.config_news import process_config
from data_loader.twitter_data_generator import TwitterDataGenerator
from data_loader.news_data_generator import NewsDataGenerator
from models.NBoW_model import NBoWModel
from trainers.NBow_trainer import NBowTrainer
from models.CNN_model import CNNModel
from trainers.CNN_trainer import CNNTrainer
from models.LSTM_model import LSTMModel
from trainers.LSTM_trainer import LSTMTrainer
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.dic_dir, config.result_dir])
    # create tensorflow session
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    # create your data generator
    # data = TwitterDataGenerator(config)
    data = TwitterDataGenerator(config)
    # create instance of the model you want
    # create tensorboard logger
    # create trainer and path all previous components to it
    if config.model == 'nbow':
        model = NBoWModel(config, data)
        logger = Logger(sess, config)
        trainer = NBowTrainer(sess, model, data, config, logger)
    elif config.model == 'cnn':
        model = CNNModel(config, data)
        logger = Logger(sess, config)
        trainer = CNNTrainer(sess, model, data, config, logger)
    elif config.model == 'lstm':
        model = LSTMModel(config, data)
        logger = Logger(sess, config)
        trainer = LSTMTrainer(sess, model, data, config, logger)
    else:
        raise NotImplementedError
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
