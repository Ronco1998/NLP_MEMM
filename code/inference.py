from preprocessing import read_test
from tqdm import tqdm


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below 
    Recursive function (use dinamic programming / memoization - find out how to do it)

    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    
    :param sentence: the sentence to tag
    :param pre_trained_weights: the weights vector
    :feature2id: the feature2id object
    :return: the predicted tags for the sentence 
             the tags sequence which maximizes the conditional probability of the tags given the sentence
    """
    B = 2  # Beam size
    n = len(sentence)  # number of words in the sentence
    m = len(feature2id.feature_statistics.tags)  # number of tags
    # Initialize the Viterbi table and backpointer table
    viterbi_table = [[0] * m for _ in range(n + 1)]
    backpointer_table = [[0] * m for _ in range(n + 1)] 

    pass #TODO: implement Viterbi function


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
