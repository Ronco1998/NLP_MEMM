from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np

# done... in main

def memm_viterbi_beam_search(sentence, pre_trained_weights, feature2id, beam_width=4):
    """
    MEMM Viterbi algorithm with beam search optimization using precomputed features.
    :param sentence: list of words (with padding at start and end)
    :param pre_trained_weights: the weights vector
    :param feature2id: the feature2id object
    :param beam_width: the beam size (number of top paths to keep at each step)
    :return: the predicted tags for the sentence
    """
    tags = list(feature2id.feature_statistics.tags)
    n = len(sentence)
    beams = [ [(('*', '*'), 0.0, [])] ]  # Each entry: (prev_tags, log_score, tag_seq)

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
                # Use precomputed features from histories_matrix if available
                if hasattr(feature2id, 'histories_matrix') and history in feature2id.histories_matrix:
                    features = feature2id.histories_matrix[history]
                else:
                    features = represent_input_with_features(history, feature2id.feature_to_idx)
                # Use log-space for scores
                log_q = sum(pre_trained_weights[f] for f in features)

                path_score = score + log_q

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
        words = sen[0][2:-1]  # Remove the two paddings at the start and the one at the end
        pred = memm_viterbi_beam_search(sen[0], pre_trained_weights, feature2id)
        pred = pred[:len(words)]  # Ensure predictions match the number of words
        for i in range(len(words)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{words[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
