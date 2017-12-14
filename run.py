from __future__ import division
from corpus_stats import get_counter

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
from neural_models import dan, cnn, lstm # noqa

import sys
import string
import re
import numpy as np

filename = 'adverbs.txt'
max_sequence_len = 60
embedding_dim = 300
percent_training = 0.9
WORD_VEC_FILE = '/Users/quinnmac/Documents/00-Documents-Archive/College Senior Year/'\
                'Semester 2/NLP/Final Project/GoogleNews-vectors-negative300.bin'
WORD_VEC_FILE = '/Users/skumar/Documents/proj/nlp/GoogleNews-vectors-negative300.bin'


def run_experiment(model_name):
    with open(filename) as input_file:
        counter = get_counter(filename)
        data = input_file.readlines()

    clusters = {}
    for i, l in enumerate(open('clusters.txt')):
        for w in l.split():
            clusters[w] = i

    # min_cutoff = 10
    # filtered_counter = {key: value for key, value in counter.iteritems() if value > min_cutoff}
    # label_set = filtered_counter.keys()
    filtered_counter = {}

    examples = []
    labels = []
    seq_lens = []
    wvModel = KeyedVectors.load_word2vec_format(WORD_VEC_FILE, binary=True)

    # Parse the data
    num_tokens = 0.0
    num_unks = 0.0
    for i, item in enumerate(data):
        label, sentence = item.split('\t')
        if label not in clusters:
            continue

        orig_sentence = sentence.decode('utf-8')
        sentence = orig_sentence.lower()
        sentence = re.sub('[' + string.punctuation + ']', '', sentence)
        # tokens = word_tokenize(sentence)
        tokens = sentence.split()

        if not tokens:
            continue

        sentence_vectors = []
        for tok in tokens:
            if tok in wvModel:
                sentence_vectors.append(wvModel[tok])
            else:
                sentence_vectors.append(np.zeros(embedding_dim))
                num_unks += 1.0
            num_tokens += 1.0

        # Compute Stylometric Features
        stylometric_features = []
        scale = float(len(orig_sentence))
        punct = string.punctuation + string.ascii_uppercase
        # punct = '?-!.'
        for c in punct:
            # stylometric_features.append(orig_sentence.count(c) / scale)
            stylometric_features.append(1 if c in orig_sentence else 0)

        # stylometric_features.append(scale)
        # s = 0.0
        # for tok in tokens:
            # s += len(tok)
        # stylometric_features.append(s/ num_tokens)

        examples.append([sentence_vectors, stylometric_features])   
        labels.append(clusters[label])
        seq_lens.append(len(tokens))
        if clusters[label] in filtered_counter:
            filtered_counter[clusters[label]] += 1
        else:
            filtered_counter[clusters[label]] = 1

    wvModel = None  # save memory
    num_examples = len(examples)
    num_labels = len(set(clusters.values()))
    print('Number of labels: ' + str(num_labels))
    print('Number of examples: ' + str(num_examples))
    print('Guessing Most Common Label: ' + str(labels.count(max(set(labels), key=labels.count)) / num_examples))

    print('Unknown Tokens: ' + str(num_unks))
    print('Total Tokens: ' + str(num_tokens))
    print('Unknown Words: ' + str(num_unks / num_tokens))

    # Partition the data
    shuffled_idxs = range(num_examples)
    np.random.seed(42)
    np.random.shuffle(shuffled_idxs)
    num_training = int(percent_training*num_examples)

    train_idxs = shuffled_idxs[:num_training]    

    label_set = filtered_counter.keys()
    max_label_value = max(filtered_counter.values())
    print(len(train_idxs))
    more_idxs = []
    for i in train_idxs:
        num_append = int(max_label_value / filtered_counter[label_set[labels[i]]]) - 1
        more_idxs.extend([i] * num_append)
    train_idxs.extend(more_idxs)
    print(len(train_idxs))

    train_examples = [examples[i] for i in train_idxs]
    train_labels = [labels[i] for i in train_idxs]


    test_idxs = shuffled_idxs[num_training:]
    test_examples = [examples[i] for i in test_idxs]
    test_labels = [labels[i] for i in test_idxs]

    # Let the fun begin
    print(WORD_VEC_FILE)
    eval(model_name)(train_examples, train_labels, test_examples, test_labels, num_labels, embedding_dim)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run.py <model>')
        sys.exit(0)

    run_experiment(sys.argv[1])
