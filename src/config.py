import datetime

# category2id = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4, 'world': 5, 'finance': 6, 'game': 7,
#                'travel': 8, 'military': 9, 'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
#                'story': 15, 'regimen': 16, 'essay': 17}
category2id = {"_red_heart_": 0, "_smiling_face_with_hearteyes_": 1, "_face_with_tears_of_joy_": 2, "_two_hearts_": 3,
               "_fire_": 4, "_smiling_face_with_smiling_eyes_": 5, "_smiling_face_with_sunglasses_": 6, "_sparkles_": 7,
               "_blue_heart_": 8, "_face_blowing_a_kiss_": 9, "_camera_": 10, "_United_States_": 11, "_sun_": 12,
               "_purple_heart_": 13, "_winking_face_": 14, "_hundred_points_": 15, "_beaming_face_with_smiling_eyes_": 16,
               "_Christmas_tree_": 17, "_camera_with_flash_": 18, "_winking_face_with_tongue_": 19}
id2category = {index: label for label, index in category2id.items()}


max_sent_len = 34
num_class = 20
max_word_len = 38

ROOT = '../data/twitter_data/English'
ES_ROOT = '../data/twitter_data/Spanish'
DATA_DIR = ROOT + '/word'
ES_DATA_DIR = ES_ROOT + '/word'
# train_file = DATA_DIR + '/train.txt'
# dev_file = DATA_DIR + '/dev.txt'
# dev_new_file = DATA_DIR + '/dev_new.txt'
# test_file = DATA_DIR + '/us_test.text'
dev_new_file = ROOT + '/processed/us_trial_new.json'
train_file = ROOT + "/processed/tweet_processed.json"
dev_file = ROOT + "/processed/us_trial.json"
train_file_final = ROOT + "/processed/2of3.json"
train_file_final2 = ROOT + "/processed/2.train"
train_file_final3 = ROOT + "/processed/3.train"
dev_file_final = ROOT + "/processed/1of3.json"
dev_file_final2 = ROOT + "/processed/2.trial"
dev_file_final3 = ROOT + "/processed/3.trial"
test_file_final = ROOT + "/processed/us_test.json"
train_balance_file = ROOT + "/processed/rebalance_tweet_processed.train"
train_es_file = ES_DATA_DIR + '/train_es.txt'
dev_es_file = ES_DATA_DIR + '/dev_es.txt'
word_embed_google = ROOT + '/embed/google_w2v.vocab.vector'
word_embed_SWM = ROOT + '/embed/SWM_NEW.vocab.vector'
word_embed_w2v = ROOT + '/embed/word2vec_300.vec'
word_embed_glove = ROOT + '/embed/Glove.vocab_300.vector'
word_embed_SSWE = ROOT + 'embed/SSWE.vocab.vector'
word_embed_SWV = ROOT + 'embed/SWV.vocab.vector'
word_dim = 300
char_dim = 50
ner_dim = 50
pos_dim = 50

OUTPUT_DIR = '../output'
MODEL_DIR = '../model'
DIC_DIR = '../dic'
FEATURE_DIR = '../feature'
SAVE_DIR = '../save'
VOCABULARY_DIR = '../vocabulary'

w2i_file = DIC_DIR + '/w2i.p'
c2i_file = DIC_DIR + '/c2i.p'
n2i_file = DIC_DIR + '/n2i.p'
p2i_file = DIC_DIR + '/p2i.p'
oov_file = DIC_DIR + '/oov.p'
we_file = DIC_DIR + '/we.p'
rf_file = DIC_DIR + '/rf2_unigram_5.txt'
top_key_file = DIC_DIR + '/top_key.txt'
dev_model_file = MODEL_DIR + '/dev_model_final_notvoerlap/'
dev_model_file1 = MODEL_DIR + '/dev_model_final_all_lr0.1/'
dev_model_file2 = MODEL_DIR + '/dev_model_final_all_dp0.7/'
dev_model_file3 = MODEL_DIR + '/dev_model_final_all_dp0.5/'
dev_model_file4 = MODEL_DIR + '/dev_model_final_all_v2/'

# dev_predict_file = OUTPUT_DIR + '/dev-predicts-final.txt'
dev_predict_file = OUTPUT_DIR + '/notoverlap_traditional.txt'
dev_predict_file1 = OUTPUT_DIR + '/dev-predicts-final-nbow.txt'
dev_predict_file2 = OUTPUT_DIR + '/dev-predicts-final-cnn.txt'
dev_predict_file3 = OUTPUT_DIR + '/dev-predicts-final-lstm.txt'
dev_predict_dense = OUTPUT_DIR + '/dense.txt'
dev_predict_sparse = OUTPUT_DIR + '/sparse.txt'
dev_predict_nlp = OUTPUT_DIR + '/dev_traditional.txt'
dev_predict_ensemble_file = OUTPUT_DIR + '/dev-ensemble-result.txt'
dev_predict_ensemble_file_final = OUTPUT_DIR + '/dev-ensemble-result-final.txt'
test_predict_file = OUTPUT_DIR + '/test-predicts-notoverlap.txt'
# test_predict_all_file = OUTPUT_DIR + '/result_2.txt'
test_predict_nlp = OUTPUT_DIR + '/result_nooverlap_traditional.txt'
test_predict_ensemble_file_final = OUTPUT_DIR + '/test-ensemble-result-final.txt'
dev_gold_new_file = DATA_DIR + '/dev_new.txt'
dev_gold_file = DATA_DIR + '/dev.txt'
dev_gold_final_file = DATA_DIR + '/dev_gold_final.txt'
# train_data_file = OUTPUT_DIR + '/train_data.txt'

VOCAB_NORMAL_WORDS_PATH = VOCABULARY_DIR + '/normal_word.pkl'
STOP_WORD_PATH = VOCABULARY_DIR + '/nltk_stopwords.txt'
NEGATION_PATH = VOCABULARY_DIR + "/negation terms.txt"

LEXI_SOURCE = "../sentiment_lexicon"
LEXI_BL = LEXI_SOURCE + "/Bing Liu/BL.lexi"
LEXI_AFINN = LEXI_SOURCE + "/AFINN/AFINN-111.lexi"
LEXI_GI = LEXI_SOURCE + "/General Inquirer/GI.lexi"
LEXI_IMDB = LEXI_SOURCE + "/imdb/IMDB.lexi"
LEXI_MPQA = LEXI_SOURCE + "/MPQA/MPQA.lexi"
LEXI_NRC140_U = LEXI_SOURCE + "/NRC140/Nunigram.lexi"
LEXI_NRC140_B = LEXI_SOURCE + "/NRC140/Nbigram.lexi"
LEXI_NRCEMOTION = LEXI_SOURCE + "/NRCEmotion/NRCEmoti.lexi"
LEXI_NRCHASHTAG_U = LEXI_SOURCE + "/NRCHashtag/Hunigram.lexi"
LEXI_NRCHASHTAG_B = LEXI_SOURCE + "/NRCHashtag/Hbigram.lexi"
LEXI_SENTIWORDNET = LEXI_SOURCE + "/SentiWordNet_3.0.0/SentiWordNet.lexi"

train_feature_file = FEATURE_DIR + "/train_feature_notoverlap_final_new.txt"
dev_feature_file = FEATURE_DIR + "/dev_feature_notoverlap_final_new.txt"
test_feature_file = FEATURE_DIR + "/test_feature_notoverlap_final_new.txt"
