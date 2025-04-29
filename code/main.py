import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test

# ----------------------------------------------------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from check_submission import compare_files
# ----------------------------------------------------------------

def main():
    threshold = 7  # or higher, experiment to get under 10,000 features
    lam = 0.1

    train_path = "data/train1.wtag"

    #test_path = "data/comp1.words" # for competition purposes
    test_path = "data/test1.wtag" # for testing purposes

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)

    #------------------------------------------------------------
    # Compute and print accuracy if test_path is a labeled file
    if 'test1.wtag' in test_path:
        acc, _ = compare_files(test_path, predictions_path)
        print(f"Token-level accuracy on test set: {acc*100:.2f}%")
    #------------------------------------------------------------

if __name__ == '__main__':
    main()
