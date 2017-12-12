# category2id = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4, 'world': 5, 'finance': 6, 'game': 7,
#                'travel': 8, 'military': 9, 'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
#                'story': 15, 'regimen': 16, 'essay': 17}
category2id = {"_red_heart_": 0, "_smiling_face_with_hearteyes_": 1, "_face_with_tears_of_joy_": 2, "_two_hearts_": 3,
               "_fire_": 4, "_smiling_face_with_smiling_eyes_": 5, "_smiling_face_with_sunglasses_": 6, "_sparkles_": 7,
               "_blue_heart_": 8, "_face_blowing_a_kiss_": 9, "_camera_": 10, "_United_States_": 11, "_sun_": 12,
               "_purple_heart_": 13, "_winking_face_": 14, "_hundred_points_": 15, "_beaming_face_with_smiling_eyes_": 16,
               "_Christmas_tree_": 17, "_camera_with_flash_": 18, "_winking_face_with_tongue_": 19}
id2category = {index: label for label, index in category2id.items()}


max_sent_len = 30
num_class = 20
max_word_len = 10

ROOT = '../data/twitter_data'

DATA_DIR = ROOT + '/word'
# train_file = DATA_DIR + '/train.txt'
# dev_file = DATA_DIR + '/dev.txt'
# test_file = DATA_DIR + '/test.txt'
train_file = ROOT + "/processed/tweet_processed.json"
dev_file = ROOT + "/processed/us_trial.json"
word_embed_google = ROOT + '/embed/google_w2v.vocab.vector'
word_embed_SWM = ROOT + '/embed/SWM.vocab.vector'
word_embed_w2v = ROOT + '/embed/word2vec_300.vec'
word_dim = 300
char_dim = 50
ner_dim = 50

OUTPUT_DIR = '../output'
MODEL_DIR = '../model'
w2i_file = OUTPUT_DIR + '/w2i.p'
c2i_file = OUTPUT_DIR + '/c2i.p'
n2i_file = OUTPUT_DIR + 'n2i.p'
we_file = OUTPUT_DIR + '/we.p'

dev_model_file = MODEL_DIR + '/dev_model'
dev_predict_file = OUTPUT_DIR + '/dev-predicts_08.txt'
test_predict_file = OUTPUT_DIR + '/test-predicts.txt'
dev_gold_file =  DATA_DIR + '/dev_gold_file.txt'
train_data_file = OUTPUT_DIR + '/train_data.txt'
SAVE_DIR = '../save'
VOCABULARY_DIR = '../vocabulary'
VOCAB_NORMAL_WORDS_PATH = VOCABULARY_DIR + '/normal_word.pkl'
