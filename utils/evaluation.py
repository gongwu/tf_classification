# coding: utf-8
import pandas as pd
from configs import config_twitter
from utils.confusion_matrix import Alphabet, ConfusionMatrix

DICT_LABEL_TO_INDEX = {"_red_heart_": 0, "_smiling_face_with_hearteyes_": 1, "_face_with_tears_of_joy_": 2, "_two_hearts_": 3,
               "_fire_": 4, "_smiling_face_with_smiling_eyes_": 5, "_smiling_face_with_sunglasses_": 6, "_sparkles_": 7,
               "_blue_heart_": 8, "_face_blowing_a_kiss_": 9, "_camera_": 10, "_United_States_": 11, "_sun_": 12,
               "_purple_heart_": 13, "_winking_face_": 14, "_hundred_points_": 15, "_beaming_face_with_smiling_eyes_": 16,
               "_Christmas_tree_": 17, "_camera_with_flash_": 18, "_winking_face_with_tongue_": 19}

DICT_INDEX_TO_LABEL = {index: label for label, index in DICT_LABEL_TO_INDEX.items()}


def confusion_matrix(gold, pred):
    labels = sorted(set(list(gold) + list(pred)), reverse=False)
    line_0 = ["%5d" % t for t in labels]
    print("Gold\\Pred\t| " + " \t".join(line_0))

    for cur in labels:
        count = dict(((l, 0) for l in labels))
        for i in range(len(gold)):
            if gold[i] != cur: continue
            count[pred[i]] += 1
        count = ["%5d" % (count[l]) for l in labels]
        print("\t%5d\t| " % cur + " \t".join(count))


def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        gold_list = [int(line.strip().split('\t')[0]) for line in gold_file]
        predicted_list = [int(line.strip().split("\t")[0]) for line in predict_file]
        predict_labels = [config_twitter.id2category[int(predict)] for predict in predicted_list]
        gold_labels = [config_twitter.id2category[int(gold)] for gold in gold_list]
        binary_alphabet = Alphabet()
        for i in range(20):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predict_labels, gold_labels)

        confusion_matrix(gold_list, predicted_list)
        cm.print_summary()
        macro_p, macro_r, macro_f1 = cm.get_average_prf()
        overall_accuracy = cm.get_accuracy()
        return overall_accuracy, macro_p, macro_r, macro_f1


def Evaluation_lst(gold_label, predict_label, print_all=False):
    binary_alphabet = Alphabet()
    for i in range(20):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)

    if print_all:
        cm.print_out()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy


def Evaluation_all(gold_label, predict_label):
    binary_alphabet = Alphabet()
    for i in range(20):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)
    macro_p, macro_r, macro_f1 = cm.get_average_prf()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy, macro_p, macro_r, macro_f1


def case_study(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        # gold_list = []
        # for line in gold_file:
        #     gold_list.append([int(line.strip().split('\t')[0]), line.strip().split('\t')[1]])
        gold_list = [int(line.strip().split('\t')[0]) for line in gold_file]
        predicted_list = [int(line.strip().split("\t")[0]) for line in predict_file]
        error_list = []
        for i in range(len(gold_list)):
            if gold_list[i][0] != predicted_list[i]:
                error_list.append(i)
        case_list = []
        for i in error_list:
            case_list.append({'gold': DICT_INDEX_TO_LABEL[gold_list[i][0]]+" "+str(gold_list[i][0]),  'content': gold_list[i][1], 'predicted': DICT_INDEX_TO_LABEL[predicted_list[i]]+" "+str(predicted_list[i])})
        case_data = pd.DataFrame(case_list)
        return case_data


# if __name__ == '__main__':
#     dev_overall_accuracy, dev_macro_p, dev_macro_r, dev_macro_f1 = Evaluation(config.dev_predict_file,
#                                                               config.dev_gold_final_file)
#     test_overall_accuracy, test_macro_p, test_macro_r, test_macro_f1 = Evaluation(config.test_predict_file,
#                                                               config.test_gold_file)
#     print('dev: overall_accuracy = {:.5f}, macro_p = {:.5f}, macro_r = {:.5f}, macro_f1 = {:.5f}\n'
#           'test: overall_accuracy = {:.5f}, macro_p = {:.5f}, macro_r = {:.5f}, macro_f1 = {:.5f}\n'
#           .format(dev_overall_accuracy, dev_macro_p, dev_macro_r, dev_macro_f1, test_overall_accuracy, test_macro_p, test_macro_r, test_macro_f1))
    # case_list = case_study(config.test_predict_file, config.test_gold_file)
    # pd.set_option('display.width', 1000)
    # print(case_list.head(10000))

