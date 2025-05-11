# competition part
from inference import tag_all_test
import pickle

# comp1
weights1_path = 'weights1.pkl'
with open(weights1_path, 'rb') as f:
        optimal_params, feature2id1 = pickle.load(f)
pre_trained_weights_1 = optimal_params[0]

comp1_path = "data/comp1.words" # for competition purposes
predictions_path_comp1 = 'comp_m1_341241297_206134867.wtag'
tag_all_test(comp1_path, pre_trained_weights_1, feature2id1, predictions_path_comp1)

# comp2
weights2_path = 'weights2.pkl'
with open(weights2_path, 'rb') as f:
        optimal_params, feature2id2 = pickle.load(f)
pre_trained_weights_2 = optimal_params[0]

comp2_path = "data/comp2.words" # for competition purposes
predictions_path_comp2 = 'comp_m2_341241297_206134867.wtag'
tag_all_test(comp2_path, pre_trained_weights_2, feature2id2, predictions_path_comp2)
