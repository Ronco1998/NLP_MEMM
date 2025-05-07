import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil

# ----------------------------------------------------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from check_submission import compare_files
# ----------------------------------------------------------------

def perform_5_fold_cross_validation(train2_path, weights2_path, threshold, lam):
    # Load the dataset
    with open(train2_path, 'r') as f:
        data = f.readlines()

    accuracies = []
    features2id = []

    # Split data into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_weights = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
        # Create train and test sets for this fold
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        # Save temporary train and test files
        train_fold_path = f"data/train2_fold{fold_idx + 1}.wtag"
        test_fold_path = f"data/test2_fold{fold_idx + 1}.wtag"
        with open(train_fold_path, 'w') as f:
            f.writelines(train_data)
        with open(test_fold_path, 'w') as f:
            f.writelines(test_data)

        # Preprocess and train on this fold using the global feature2id
        statistics, feature2id_fold = preprocess_train(train_fold_path, threshold)
        features2id.append(feature2id_fold)
        weights_path = f"weights_fold{fold_idx + 1}.pkl"
        get_optimal_vector(statistics=statistics, feature2id=feature2id_fold, weights_path=weights_path, lam=lam)

        # Load the weights for this fold
        with open(weights_path, 'rb') as f:
            optimal_params, _ = pickle.load(f)
        pre_trained_weights = optimal_params[0]
    
        print(pre_trained_weights)
        prediction_fold_path = f'predictions_fold{fold_idx + 1}.wtag'
        tag_all_test(test_fold_path, pre_trained_weights, feature2id_fold, prediction_fold_path)

        #------------------------------------------------------------
        # Compute and print accuracy if test_path is a labeled file
        if 'test2' in test_fold_path:
            acc, _ = compare_files(test_fold_path, prediction_fold_path)
            accuracies.append(acc)
            print(f'Model 2 - test accuracy in fold {fold_idx + 1}:')
            print(f"Token-level accuracy on test set: {acc*100:.2f}%")
        #------------------------------------------------------------
            
        fold_weights.append(optimal_params[0])  # Collect weights
          
    # get the index of the best fold
    best_fold_idx = np.argmax(accuracies)
    best_feature2id = features2id[best_fold_idx]
    # Save the averaged weights to the final weights file using the global feature2id
    with open(weights2_path, 'wb') as f:
        pickle.dump(fold_weights[best_fold_idx], f)

    return weights2_path, best_feature2id

def main():
    # threshold = 8  # or higher, experiment to get under 10,000 features
    threshold_m1 = {"f100": 12, "f101": 8, "f102": 8, "f103": 8, "f104": 10, "f105": 7, "f106": 8, "f107": 8,
                    "f_number": 6, "f_Capital": 4, "f_apostrophe": np.inf, "f_plural": 6, "f_bio_pre_suf": np.inf, "f_hyfen": 8,
                    "f_econ_terms": 8, "f_bio_terms": np.inf, "f_CapCap": 7, "f_CapCapCap": 7, "f_allCap": 7,
                    "f_dot": 8}
    threshold_m2 = {"f100": 7, "f101": 8, "f102": 9, "f103": 6, "f104": 10, "f105": 7, "f106": 8, "f107": np.inf,
                    "f_number": 6, "f_Capital": 4, "f_apostrophe": np.inf, "f_plural": 10, "f_bio_pre_suf": 8, "f_hyfen": 7,
                    "f_econ_terms": np.inf, "f_bio_terms": 8, "f_CapCap": np.inf, "f_CapCapCap": np.inf, "f_allCap": 5,
                    "f_dot": 7}

    lam = 0.1

    # model 1
    train1_path = "data/train1.wtag"
    test_train_1_path = "data/train_test1.wtag" # for testing purposes
    shutil.copy(train1_path, test_train_1_path)
    test_path = "data/test1.wtag" # for testing purposes
    weights1_path = 'weights1.pkl'
    predictions1_path = 'predictions1.wtag'
    
    statistics, feature2id1 = preprocess_train(train1_path, threshold_m1)
    get_optimal_vector(statistics = statistics, feature2id = feature2id1, weights_path = weights1_path, lam=lam)

    with open(weights1_path, 'rb') as f:
        optimal_params, feature2id1 = pickle.load(f)
    pre_trained_weights_1 = optimal_params[0]
    
    print(pre_trained_weights_1)
    tag_all_test(test_path, pre_trained_weights_1, feature2id1, predictions1_path)

    #------------------------------------------------------------
    # Compute and print accuracy if test_path is a labeled file
    if 'test1.wtag' in test_path:
        acc_test1, _ = compare_files(test_path, predictions1_path)
        print(f'Model 1 - test accuracy')
        print(f"Token-level accuracy on test set: {acc_test1*100:.2f}%")
    #------------------------------------------------------------

    predictions1_on_train_path = 'predictions1_on_train.wtag'
    tag_all_test(test_train_1_path, pre_trained_weights_1, feature2id1, predictions1_on_train_path)
    #------------------------------------------------------------
    # Compute and print accuracy if train_path is a labeled file
    if 'test1.wtag' in test_train_1_path:
        acc_train1, _ = compare_files(test_train_1_path, predictions1_on_train_path)
        print(f'Model 1 - train accuracy')
        print(f"Token-level accuracy on train set: {acc_train1*100:.2f}%")
    #------------------------------------------------------------
    # compute and print confusion matrix
    #TODO: print confusion matrix

    # model 2 -> 5-fold cross-validation of train2.wtag to find a good model
    train2_path = "data/train2.wtag"
    # creating a copy of the train2.wtag for testing purposes (only for us)
    test_on_train_path = "data/test2.wtag" 
    shutil.copy(train2_path, test_on_train_path)

    weights2_path = 'weights2.pkl'
    _, feature2id2 = perform_5_fold_cross_validation(train2_path, weights2_path, threshold_m2, lam)

    # After cross-validation, use the averaged weights for tagging
    predictions2_on_train_path = 'predictions2_on_train.wtag'

    # Ensure the weights2_path and feature2id2 are correctly used for tagging
    with open(weights2_path, 'rb') as f:
        optimal_params2 = pickle.load(f)
    pre_trained_weights_2 = optimal_params2  # Extract weights

    # Use the best weights and feature2id for tagging
    tag_all_test(test_on_train_path, pre_trained_weights_2, feature2id2, predictions2_on_train_path)

    # Compute and print accuracy for the train2.wtag dataset
    if 'test2.wtag' in test_on_train_path:
        acc_train2, _ = compare_files(test_on_train_path, predictions2_on_train_path)
        print(f'Model 2 - train accuracy')
        print(f"Token-level accuracy on train set: {acc_train2*100:.2f}%")


    # # competition part
    # # comp1
    # comp1_path = "data/comp1.words" # for competition purposes
    # predictions_path_comp1 = 'comp_m1_341241297_206134867.wtag'
    # tag_all_test(comp1_path, pre_trained_weights_1, feature2id, predictions_path_comp1)

    # # comp2
    # comp2_path = "data/comp2.words" # for competition purposes
    # predictions_path_comp2 = 'comp_m2_341241297_206134867.wtag'
    # tag_all_test(comp2_path, pre_trained_weights_2, feature2id, predictions_path_comp2)
    

if __name__ == '__main__':
    main()
