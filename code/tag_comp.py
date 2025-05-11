import nltk
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# download once, if you haven't already
# nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def tag_comp1(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tokens = nltk.word_tokenize(line.strip())
            tagged = nltk.pos_tag(tokens)
            # join back into a single line with word_TAG tokens
            f_out.write(' '.join(f"{w}_{t}" for w,t in tagged) + '\n')

# Example usage
input_file1 = "data/comp1.words"
output_file1 = "data/comp1_tagged.wtag"
tag_comp1(input_file1, output_file1)

input_file2 = "data/comp2.words"
output_file2 = "data/comp2_tagged.wtag"
tag_comp1(input_file2, output_file2)