from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "hundred", "thousand", "million", "billion"}
# added known prefixes and suffixes for biology - can add\edit acording to outside knowledge
prefixes_bio = {"ana", "angio", "arthr", "arthro", "auto", "blast", "cephal", "cephalo", "chrom", "chromo",
                "cyto", "dactyl", "diplo", "ect", "ecto", "end", "endo", "epi", "erythr", "erythro",
                "ex", "exo", "eu", "glyco", "gluco", "haplo", "hem", "hemo", "hemato", "heter", "hetero",
                "karyo", "caryo", "meso", "my", "myo", "peri", "phag", "phago", "poly", "proto", "staphyl",
                "staphylo", "tel", "telo", "zo", "zoo"}
suffixes_bio = {"ase", "cyte" "derm", "dermis", "ectomy", "stomy", "emia", "aemia", "genic", "itis", "kinesis",
                "kinesia", "lysis", "oma", "osis", "otic", "otomy", "tomy", "penia", "phage", "phagia",
                "phile", "philic", "plasm", "plasmo", "scope", "stasis", "troph", "trophy"}

class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107",
                              "f_number", "f_Capital", "f_apostrophe", "f_plural", "f_bio"]  # added f_plural + f_bio
        #TODO: why f_apostrophe?
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag) # adding all different tags to the set
                    self.tags_counts[cur_tag] += 1 # counting the number of times each tag appeared
                    self.words_count[cur_word] += 1 # counting the number of times each word appeared in the text

                    self.check_all_features(self.feature_rep_dict, cur_word, cur_tag, word_idx, split_words)

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def check_feature_f100(self, cur_word, cur_tag):
        return True  # f100 applies to all word-tag pairs

    def check_feature_f101(self, cur_word, cur_tag):
        known_suffixes = {
            'ing': 'VBG',
            'ed': 'VBD',
            'ly': 'RB',
            's': 'NNS',
            'es': 'VBZ',
            'ion': 'NN',
            'ions': 'NNS',
            'able': 'JJ',
            'ible': 'JJ', 
            'ic': 'JJ', 
            'ical': 'JJ'
        }
        for suffix, tag in known_suffixes.items():
            if cur_word.endswith(suffix) and cur_tag == tag:
                return True
        return False

    def check_feature_f102(self, cur_word, cur_tag):
        known_prefixes = {
            'un': 'JJ',
            'pre': 'NN',
            'pre': 'JJ',
            're': 'VB',
            'dis': 'VB',
            'dis': 'JJ',
            'in': 'JJ',
            'mis': 'VB'
        }
        for prefix, tag in known_prefixes.items():
            if cur_word.startswith(prefix) and cur_tag == tag:
                return True
        return False

    def check_feature_f103(self, word_idx, split_words, cur_tag): # the word had two previous words
        # Use all observed trigrams (prev2_tag, prev1_tag, cur_tag)
        return word_idx >= 2

    def check_feature_f104(self, word_idx, split_words, cur_tag): # the word had one previous word
        # Use all observed bigrams (prev1_tag, cur_tag)
        return word_idx >= 1

    def check_feature_f105(self, cur_tag):
        return True  # f105 applies to all tags

    def check_feature_f106(self, word_idx, split_words, cur_tag):
        # Use all observed (prev_word, cur_tag)
        return word_idx >= 1

    def check_feature_f107(self, word_idx, split_words, cur_tag):
        # Use all observed (next_word, cur_tag)
        return word_idx < len(split_words) - 1

    def check_feature_f_number(self, cur_word, cur_tag, word_idx=None, split_words=None):
        return any(char.isdigit() for char in cur_word) or cur_word.lower() in NUMBER_WORDS

    def check_feature_f_Capital(self, cur_word, cur_tag):
        # Fires if the word starts with a capital letter
        return cur_word and cur_word[0].isupper()

    def check_feature_f_apostrophe(self, cur_word, cur_tag):
        return "'" in cur_word

    def check_feature_f_plural(self, cur_word, cur_tag):
        # Fires if the word is likely plural (simple heuristic)
        return cur_word.lower().endswith('s') and cur_tag in {"NNS", "NNPS"}
    
    def check_feature_f_bio(self, cur_word, cur_tag):
        # Check if the word starts with a known prefix or ends with a known suffix
        for prefix in prefixes_bio:
            if cur_word.startswith(prefix) and cur_tag in {"NN", "NNS"}:
                return True
        for suffix in suffixes_bio:
            if cur_word.endswith(suffix) and cur_tag in {"NN", "NNS"}:
                return True
        return False

    def check_all_features(self, feature_rep_dict, cur_word, cur_tag, word_idx, split_words):

        if self.check_feature_f100(cur_word, cur_tag):
            feature_rep_dict["f100"][(cur_word, cur_tag)] = feature_rep_dict["f100"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f101(cur_word, cur_tag):
            feature_rep_dict["f101"][(cur_word, cur_tag)] = feature_rep_dict["f101"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f102(cur_word, cur_tag):
            feature_rep_dict["f102"][(cur_word, cur_tag)] = feature_rep_dict["f102"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f103(word_idx, split_words, cur_tag): 
            feature_rep_dict["f103"][(split_words[word_idx - 2].split('_')[1], split_words[word_idx - 1].split('_')[1], cur_tag)] = feature_rep_dict["f103"].get((split_words[word_idx - 2].split('_')[1], split_words[word_idx - 1].split('_')[1], cur_tag), 0) + 1
        if self.check_feature_f104(word_idx, split_words, cur_tag):
            feature_rep_dict["f104"][(split_words[word_idx - 1].split('_')[1], cur_tag)] = feature_rep_dict["f104"].get((split_words[word_idx - 1].split('_')[1], cur_tag), 0) + 1
        if self.check_feature_f105(cur_tag):
            feature_rep_dict["f105"][(cur_tag,)] = feature_rep_dict["f105"].get((cur_tag,), 0) + 1
        if self.check_feature_f106(word_idx, split_words, cur_tag):
            feature_rep_dict["f106"][(split_words[word_idx - 1].split('_')[0], cur_tag)] = feature_rep_dict["f106"].get((split_words[word_idx - 1].split('_')[0], cur_tag), 0) + 1
        if self.check_feature_f107(word_idx, split_words, cur_tag):
            feature_rep_dict["f107"][(split_words[word_idx + 1].split('_')[0], cur_tag)] = feature_rep_dict["f107"].get((split_words[word_idx + 1].split('_')[0], cur_tag), 0) + 1
        if self.check_feature_f_number(cur_word, cur_tag, word_idx, split_words):
            feature_rep_dict["f_number"][(cur_word, cur_tag)] = feature_rep_dict["f_number"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f_Capital(cur_word, cur_tag):
            feature_rep_dict["f_Capital"][(cur_word, cur_tag)] = feature_rep_dict["f_Capital"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f_apostrophe(cur_word, cur_tag):
            feature_rep_dict["f_apostrophe"][(cur_word, cur_tag)] = feature_rep_dict["f_apostrophe"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f_plural(cur_word, cur_tag):
            feature_rep_dict["f_plural"][(cur_word, cur_tag)] = feature_rep_dict["f_plural"].get((cur_word, cur_tag), 0) + 1
        if self.check_feature_f_bio(cur_word, cur_tag):
            feature_rep_dict["f_bio"][(cur_word, cur_tag)] = feature_rep_dict["f_bio"].get((cur_word, cur_tag), 0) + 1


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        # when self.threshold is a dicionary, each feature family has its own threshold

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries 

        self.feature_to_idx = {
            "f100": OrderedDict(),
            "f101": OrderedDict(),
            "f102": OrderedDict(),
            "f103": OrderedDict(),
            "f104": OrderedDict(),
            "f105": OrderedDict(),
            "f106": OrderedDict(),
            "f107": OrderedDict(),
            "f_number": OrderedDict(),
            "f_Capital": OrderedDict(),
            # "f_apostrophe": OrderedDict(),
            "f_plural": OrderedDict(),
            "f_bio": OrderedDict()
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        # use the threshholds for each model
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold[feat_class]:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

        # After assigning indices, precompute and cache features for all histories
        for hist in self.feature_statistics.histories:
            features = represent_input_with_features(hist, self.feature_to_idx)
            self.represent_input_with_features[hist] = features
            self.histories_matrix[hist] = features  # Cache in histories_matrix as well

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            # small_r - which row (history) we are in the small matrix
            # big_r - which row (history) we are in the big matrix
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) -> List[int]:
    """
    Extract feature vector for a given history using precomputed features if available.
    @param history: tuple{current_word, current_tag, previous_word, previous_tag, pre_previous_word, pre_previous_tag, next_word}
    @param dict_of_dicts: a dictionary of each feature and the index it was given
    @return a list with all features that are relevant to the given history
    """
    # Use precomputed features if available
    if hasattr(dict_of_dicts, 'histories_features') and history in dict_of_dicts.histories_features:
        return dict_of_dicts.histories_features[history]

    c_word = history[0]
    c_tag = history[1]
    features = []

    # f100: current word is X and current tag is T
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f101: prefix of current word (up to 4 characters) is X and current tag is T
    for prefix_length in range(1, 5):
        prefix = c_word[:prefix_length]
        if (prefix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(prefix, c_tag)])

    # f102: suffix of current word (up to 4 characters) is X and current tag is T
    for suffix_length in range(1, 5):
        suffix = c_word[-suffix_length:]
        if (suffix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(suffix, c_tag)])

    # f103: previous two tags are (X, Y) and current tag is T
    if (history[5], history[3], c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(history[5], history[3], c_tag)])

    # f104: previous tag is X and current tag is T
    if (history[3], c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(history[3], c_tag)])

    # f105: current tag is T
    if (c_tag,) in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][(c_tag,)])

    # f106: previous word is X and current tag is T
    if (history[2], c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(history[2], c_tag)])

    # f107: next word is X and current tag is T
    if (history[6], c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(history[6], c_tag)])

    # f_number: fires if the word contains a digit or is a number word and current tag is T
    if any(char.isdigit() for char in c_word) or c_word.lower() in NUMBER_WORDS:
        if (c_word, c_tag) in dict_of_dicts.get("f_number", {}):
            features.append(dict_of_dicts["f_number"][(c_word, c_tag)])

    # f_Capital: fires if the word starts with a capital letter and current tag is T
    if c_word and c_word[0].isupper():
        if (c_word, c_tag) in dict_of_dicts.get("f_Capital", {}):
            features.append(dict_of_dicts["f_Capital"][(c_word, c_tag)])

    # f_apostrophe: fires if the word contains an apostrophe and current tag is T
    if "'" in c_word:
        if (c_word, c_tag) in dict_of_dicts.get("f_apostrophe", {}):
            features.append(dict_of_dicts["f_apostrophe"][(c_word, c_tag)])

    # f_plural: fires if the word is likely plural and current tag is NNS or NNPS
    if (c_word, c_tag) in dict_of_dicts.get("f_plural", {}):
        features.append(dict_of_dicts["f_plural"][(c_word, c_tag)])

    # f_bio: fires if the word starts with a known prefix or ends with a known suffix and current tag is NNS or NNPS
    if (c_word, c_tag) in dict_of_dicts.get("f_bio", {}):
        features.append(dict_of_dicts["f_bio"][(c_word, c_tag)])
    #TODO: check if this does what its supposed to

    return features

def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
