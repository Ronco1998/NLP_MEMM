import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
import numpy as np
import random

# Perform the equivalent of shutil.copy using basic file operations

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

def perform_k_fold_cross_validation(k, train2_path, weights2_path, threshold, lam):
    # Load the dataset
    with open(train2_path, 'r') as f:
        data = f.readlines()

    # Shuffle the data to ensure randomness
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Split data into 5 folds
    num_folds = k
    fold_size = len(shuffled_data) // num_folds
    folds = [shuffled_data[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    if len(shuffled_data) % num_folds != 0:
        # Add remaining data to the last fold
        folds[-1].extend(shuffled_data[num_folds * fold_size:])

    accuracies = []
    fold_opt_params = []
    fold_feature2id = []

    for fold_idx in range(num_folds):
        # Create train and test sets for this fold
        test_data = folds[fold_idx]
        train_data = [item for i, fold in enumerate(folds) if i != fold_idx for item in fold]

        # Save temporary train and test files
        train_fold_path = f"data/train2_fold{fold_idx + 1}.wtag"
        test_fold_path = f"data/test2_fold{fold_idx + 1}.wtag"
        with open(train_fold_path, 'w') as f:
            f.writelines(train_data)
        with open(test_fold_path, 'w') as f:
            f.writelines(test_data)

        # Preprocess and train on this fold using the global feature2id
        statistics, feature2id_fold = preprocess_train(train_fold_path, threshold)
        weights_path = f"weights_fold{fold_idx + 1}.pkl"
        get_optimal_vector(statistics=statistics, feature2id=feature2id_fold, weights_path=weights_path, lam=lam)

        # Load the weights for this fold
        with open(weights_path, 'rb') as f:
            optimal_params, feature2id_fold = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        fold_opt_params.append(optimal_params)
        fold_feature2id.append(feature2id_fold)

        prediction_fold_path = f'predictions_fold{fold_idx + 1}.wtag'
        tag_all_test(test_fold_path, pre_trained_weights, feature2id_fold, prediction_fold_path)

        # Compute and print accuracy if test_path is a labeled file
        if 'test2' in test_fold_path:
            acc, _ = compare_files(test_fold_path, prediction_fold_path)
            accuracies.append(acc)
            print(f'Model 2 - test accuracy in fold {fold_idx + 1}:')
            print(f"Token-level accuracy on test set: {acc * 100:.2f}%")

    print(f'Fold accuracies: {accuracies}')

    # Get the index of the best fold
    best_fold_idx = np.argmax(accuracies)
    best_optimal_params = fold_opt_params[best_fold_idx]
    best_feature2id = fold_feature2id[best_fold_idx]

    with open(weights2_path, 'wb+') as f:
        pickle.dump((best_optimal_params, best_feature2id), f)
    
    # Clean up temporary files
    for fold_idx in range(num_folds):
        train_fold_path = f"data/train2_fold{fold_idx + 1}.wtag"
        test_fold_path = f"data/test2_fold{fold_idx + 1}.wtag"
        prediction_fold_path = f'predictions_fold{fold_idx + 1}.wtag'
        weight_fold_path = f"weights_fold{fold_idx + 1}.pkl"
        if os.path.exists(weight_fold_path):
            os.remove(weight_fold_path)
        if os.path.exists(prediction_fold_path):
            os.remove(prediction_fold_path)
        if os.path.exists(train_fold_path):
            os.remove(train_fold_path)
        if os.path.exists(test_fold_path):
            os.remove(test_fold_path)

def main():
    # threshold = 8  # or higher, experiment to get under 10,000 features
    threshold_m1 = {"f100": 7, "f101": 7, "f102": 7, "f103": 7, "f104": 9, "f105": 9, "f106": 20, "f107": 20,
                    "f_number": 3, "f_Capital": 2, "f_plural": 3, "f_bio_pre_suf": np.inf, "f_hyfen": 4,
                    "f_econ_terms": 1, "f_bio_terms": np.inf, "f_CapCap": 2, "f_CapCapCap": 2, "f_allCap": 3,
                    "f_dot": 2}
    threshold_m2 = {"f100": 2, "f101": 2, "f102": 3, "f103": 3, "f104": 3, "f105": 4, "f106": 4, "f107": 4,
                    "f_number": 4, "f_Capital": 2, "f_plural": 4, "f_bio_pre_suf": 1, "f_hyfen": 2,
                    "f_econ_terms": np.inf, "f_bio_terms": 1, "f_CapCap": 2, "f_CapCapCap": 2, "f_allCap": 1,
                    "f_dot": 2}

    lam1 = 0.1
    lam2 = 0.001

    # model 1
    train1_path = "data/train1.wtag"
    test_path = "data/test1.wtag" # for testing purposes
    weights1_path = 'weights1.pkl'
    predictions1_path = 'predictions1.wtag'
    
    statistics, feature2id1 = preprocess_train(train1_path, threshold_m1)
    get_optimal_vector(statistics = statistics, feature2id = feature2id1, weights_path = weights1_path, lam=lam1)

    with open(weights1_path, 'rb') as f:
        optimal_params, feature2id1 = pickle.load(f)
    pre_trained_weights_1 = optimal_params[0]
    
    print(pre_trained_weights_1)
    tag_all_test(test_path, pre_trained_weights_1, feature2id1, predictions1_path)

    # -----------------------------------------------------------------------------------------------------------------------

    # model 2 -> 5-fold cross-validation of train2.wtag to find a good model
    train2_path = "data/train2.wtag"
    weights2_path = 'weights2.pkl'
    k = 5 # how many folds to use for cross-validation
    perform_k_fold_cross_validation(k, train2_path, weights2_path, threshold_m2, lam2)

    # Ensure the weights2_path and feature2id2 are correctly used for tagging
    with open(weights2_path, 'rb') as f:
        optimal_params2, feature2id2 = pickle.load(f)
    pre_trained_weights_2 = optimal_params2[0]  # Extract weights

    # -----------------------------------------------------------------------------------------------------------------------

    # competition part
    # creating the tagged files for submission!

    # comp1
    comp1_path = "data/comp1.words" # for competition purposes
    predictions_path_comp1 = 'comp_m1_341241297_206134867.wtag'
    tag_all_test(comp1_path, pre_trained_weights_1, feature2id1, predictions_path_comp1)

    # comp2
    comp2_path = "data/comp2.words" # for competition purposes
    predictions_path_comp2 = 'comp_m2_341241297_206134867.wtag'
    tag_all_test(comp2_path, pre_trained_weights_2, feature2id2, predictions_path_comp2)


if __name__ == '__main__':
    main()
