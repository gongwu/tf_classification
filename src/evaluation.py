# coding: utf-8

from confusion_matrix import Alphabet, ConfusionMatrix

DICT_LABEL_TO_INDEX = {"_red_heart_": 0, "_smiling_face_with_hearteyes_": 1, "_face_with_tears_of_joy_": 2, "_two_hearts_": 3,
               "_fire_": 4, "_smiling_face_with_smiling_eyes_": 5, "_smiling_face_with_sunglasses_": 6, "_sparkles_": 7,
               "_blue_heart_": 8, "_face_blowing_a_kiss_": 9, "_camera_": 10, "_United_States_": 11, "_sun_": 12,
               "_purple_heart_": 13, "_winking_face_": 14, "_hundred_points_": 15, "_beaming_face_with_smiling_eyes_": 16,
               "_Christmas_tree_": 17, "_camera_with_flash_": 18, "_winking_face_with_tongue_": 19}

DICT_INDEX_TO_LABEL = {index:label for label, index in DICT_LABEL_TO_INDEX.items()}


def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:

        gold_list = [ line.strip().split('\t')[0] for line in gold_file]
        predicted_list = [line.strip().split("\t#\t")[0] for line in predict_file]

        binary_alphabet = Alphabet()
        for i in range(20):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predicted_list, gold_list)

        cm.print_out()
        macro_p, macro_r, macro_f1 = cm.get_average_prf()
        overall_accuracy = cm.get_accuracy()
        return overall_accuracy, macro_p, macro_r, macro_f1


def Evalation_lst(gold_label, predict_label, print_all=False):
    binary_alphabet = Alphabet()
    for i in range(20):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)

    if print_all:
        cm.print_out()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy

if __name__ == '__main__':
    pass