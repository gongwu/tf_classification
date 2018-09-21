import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, config, data):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.data = data

        self.seq_len = self.config.max_sent_len
        self.embed_size = self.config.word_dim
        self.num_class = self.config.num_class
        # Add PlaceHolder
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sent_len]
        self.input_x_len = tf.placeholder(tf.int32, (None,))
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))
        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        # Add Word Embedding
        self.we = tf.Variable(self.data.embed, name='emb')

    # save function that save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.cur_epoch_tensor)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
