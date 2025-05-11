import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import zipfile
import shutil
import pickle

OUTPUT_DIRECTORY_PATH = "your_unzip_submission"
COMP_FILES_PATH = 'comps files'
MODELS_PATH = f'{OUTPUT_DIRECTORY_PATH}/trained_models'
ID1 = input("Insert ID1: ")
ID2 = input("Insert ID2 (press ENTER if submitting alone):")

def unzip_directory(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

if not os.path.exists(COMP_FILES_PATH):
    os.makedirs(COMP_FILES_PATH)
    id_suffix = ID1
    if ID2:
        id_suffix += f'_{ID2}'
    os.makedirs(f"{COMP_FILES_PATH}/{id_suffix}")

def compare_files(true_file, pred_file):
    with open(true_file, 'r') as f:
        true_data = [x.strip() for x in f.readlines() if x != '']
    with open(pred_file, 'r') as f:
        pred_data = [x.strip() for x in f.readlines() if x != '']
    if len(pred_data) != len(true_data):
        if len(pred_data) > len(true_data):
            pred_data = pred_data[:len(true_data)]
        else:
            raise KeyError
    num_correct, num_total = 0, 0
    prob_sent = set()
    predictions, true_labels = [], []
    for idx, sen in enumerate(true_data):
        pred_sen = pred_data[idx]
        if pred_sen.endswith('._.') and not pred_sen.endswith(' ._.'):
            pred_sen = pred_sen[:-3] + ' ._.'
        true_words = [x.split('_')[0] for x in sen.split()]
        true_tags = [x.split('_')[1] for x in sen.split()]
        true_labels += true_tags
        pred_words = [x.split('_')[0] for x in pred_sen.split()]
        try:
            pred_tags = [x.split('_')[1] for x in pred_sen.split()]
            predictions += pred_tags
        except IndexError:
            prob_sent.add(idx)
            pred_tags = []
            for x in pred_sen.split():
                if '_' in x:
                    pred_tags.append(x.split('_'))
                else:
                    pred_tags.append(None)
        if pred_words[-1] == '~':
            pred_words = pred_words[:-1]
            pred_tags = pred_tags[:-1]
        if pred_words != true_words:
            prob_sent.add(idx)
        elif len(pred_tags) != len(true_tags):
            prob_sent.add(idx)
        for i, (tt, tw) in enumerate(zip(true_tags, true_words)):
            num_total += 1
            if len(pred_words) > i:
                pw = pred_words[i]
                pt = pred_tags[i]
            else:
                prob_sent.add(idx)
                continue
            if pw != tw:
                continue
            if tt == pt:
                num_correct += 1
        pass
    labels = sorted(list(set(true_labels)))
    if len(prob_sent) > 0:
        print(prob_sent)

    return num_correct / num_total, prob_sent


def calc_scores(e):
    scores = []
    for sub in os.listdir(COMP_FILES_PATH):
        # print(sub)
        cur_dir = f'{COMP_FILES_PATH}/{sub}'
        comp1_file = [x for x in os.listdir(cur_dir) if x.startswith('comp_m1')]
        comp2_file = [x for x in os.listdir(cur_dir) if x.startswith('comp_m2')]
        prob1, prob2, ids = set(), set(), []
        if len(comp1_file) != 1:
            print(f'{sub} has a problem with m1!')
            e = True
            comp1 = None
        else:
            ids = comp1_file[0].replace('comp_m1_', '').split('.')[0].split('_')
            comp1_file = f'{cur_dir}/{comp1_file[0]}'
            #comp1, prob1 = compare_files('data/comp1.wtag', comp1_file)
            #comp1 = round(comp1 * 100, 2)
            comp1 = np.round(np.random.uniform(10, 100), 2)
        if len(comp2_file) != 1:
            print(f'{sub} has a problem with m2!')
            e = True
            comp2 = None
        else:
            ids = comp2_file[0].replace('comp_m2_', '').split('.')[0].split('_')
            comp2_file = f'{cur_dir}/{comp2_file[0]}'
            #comp2, prob2 = compare_files('data/comp2.wtag', comp2_file)
            #comp2 = round(comp2 * 100, 2)
            comp2 = np.round(np.random.uniform(10, 100), 2)

        cur_score = {f'ID {idx + 1}': cur_id for idx, cur_id in enumerate(ids)}
        cur_score['Comp 1 Acc'] = comp1
        cur_score['Comp 2 Acc'] = comp2
        print("Fake score for comp1: "+str(comp1))
        print("Fake score for comp2: "+str(comp2))
        if comp1 and comp2 and float(comp1) + float(comp2) < 10:
            print("Something wrong with your comp files.")
        else:
            if not e:
                print("It looks like you are ready to submit!")
        scores.append(cur_score)
        if len(prob1) > 0:
            print(comp1_file, comp1)
        if len(prob2) > 0:
            print(comp2_file, comp2)
    scores = pd.DataFrame(scores)
    scores.to_csv('scores.csv')

def check_model_size(e):
    curr_model = f"{MODELS_PATH}/weights_1.pkl"
    print(curr_model)
    try:
        with open(curr_model, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
            print(f"Number of features for model 1: {feature2id.n_total_features}")
            if feature2id.n_total_features > 10000:
                print("Model 1 exceeded the model size limit. The limit is 10,000.")
                e = True
    except:
        print("Model 1 does not exist or is corrupted.")
        e = True

    curr_model = f"{MODELS_PATH}/weights_2.pkl"
    try:
        with open(curr_model, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
            print(f"Number of features for model 2: {feature2id.n_total_features}")
            if feature2id.n_total_features > 5000:
                print("Model 2 exceeded the model size limit. The limit is 5,000.")
                e = True
    except:
        print("Model 2 does not exist or is corrupted.")
        e = True

    return e

def open_zip():
    id_suffix = ID1
    if ID2:
        id_suffix += f'_{ID2}'
    errors = False
    zip_file_path = f"HW1_{id_suffix}.zip"
    if zip_file_path not in os.listdir():
        print(f"{zip_file_path} does not exists.")
        return True
    unzip_directory(zip_file_path, OUTPUT_DIRECTORY_PATH)
    dir_files = os.listdir(OUTPUT_DIRECTORY_PATH)
    if len(dir_files) > 5:
        print("The submission contains redundant files.")
        errors = True
    comp_files = [f"comp_m1_{id_suffix}.wtag", f"comp_m2_{id_suffix}.wtag"]
    req_files = [f"report_{id_suffix}.pdf", "code", "trained_models"] + comp_files
    req_code_files = ["generate_comp_tagged.py", "main.py"]
    for file in req_files:
        if file not in dir_files:
            print(f"{file} does not exist.")
            errors = True
    code_dir_files = os.listdir(f"{OUTPUT_DIRECTORY_PATH}/code")
    for file in req_code_files:
        if file not in code_dir_files:
            print(f"{file} does not exist in your code files.")
            errors = True
    for file in comp_files:
        shutil.copy(os.path.join(OUTPUT_DIRECTORY_PATH, file), os.path.join(COMP_FILES_PATH, f"{id_suffix}"))
    return errors

if __name__ == '__main__':
    e = open_zip()
    e = check_model_size(e)
    calc_scores(e)
