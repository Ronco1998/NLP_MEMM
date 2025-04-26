from preprocessing import read_test
from tqdm import tqdm

# TODO: need to test by Accuracy for train1 and test1, for train2 there is no test set so we want to train by the Cross Validation

def memm_viterbi_beam_search(sentence, pre_trained_weights, feature2id, beam_width=3):
    """
    MEMM Viterbi algorithm with beam search optimization.
    :param sentence: list of words (with padding at start and end)
    :param pre_trained_weights: the weights vector
    :param feature2id: the feature2id object
    :param beam_width: the beam size (number of top paths to keep at each step)
    :return: the predicted tags for the sentence
    """
    tags = list(feature2id.feature_statistics.tags)
    n = len(sentence)
    beams = [ [(("*", "*"), 0.0, [])] ]  # Each entry: (prev_tags, score, tag_seq)

    for i in range(2, n-1):  # skip padding
        new_beam = []
        for prev_tags, score, tag_seq in beams[-1]:
            for t in tags:
                history = (
                    sentence[i], t,
                    sentence[i-1], prev_tags[1],
                    sentence[i-2], prev_tags[0],
                    sentence[i+1]
                )
                features = feature2id.histories_features.get(history, [])
                path_score = score + sum(pre_trained_weights[f] for f in features)
                new_beam.append(((prev_tags[1], t), path_score, tag_seq + [t]))
        # Keep only top beam_width paths
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beams.append(new_beam[:beam_width])

    # Get the best tag sequence from the last beam
    best_seq = max(beams[-1], key=lambda x: x[1])[2]
    return best_seq


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi_beam_search(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
