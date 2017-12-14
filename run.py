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
embedding_dim = 300
percent_training = 0.9
WORD_VEC_FILE = '/Users/quinnmac/Documents/00-Documents-Archive/College Senior Year/'\
                'Semester 2/NLP/Final Project/GoogleNews-vectors-negative300.bin'
WORD_VEC_FILE = '/Users/skumar/Documents/proj/nlp/GoogleNews-vectors-negative300.bin'


def run_experiment(model_name, balance_training=True):
    with open(filename) as input_file:
        # counter = get_counter(filename)
        data = input_file.readlines()

    clusters = {}
    for i, l in enumerate(open('clusters.txt')):
        for w in l.split():
            clusters[w] = i

    # min_cutoff = 10
    # filtered_counter = {key: value for key, value in counter.iteritems() if value > min_cutoff}
    # label_set = filtered_counter.keys()

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
        for c in string.punctuation:
            stylometric_features.append(orig_sentence.count(c) / scale)

        examples.append([sentence_vectors, stylometric_features])
        labels.append(clusters[label])
        seq_lens.append(len(tokens))

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

    if balance_training:
        print('Num training examples unbalanced: ' + str(len(train_idxs)))
        train_counter = {}
        for i in train_idxs:
            label = labels[i]
            train_counter[label] = train_counter.get(label, 0.0) + 1.0

        max_label_value = max(train_counter.values())
        print('Max training label frequency: ' + str(max_label_value))
        more_idxs = []
        for i in train_idxs:
            num_label = train_counter[labels[i]]
            factor = int(max_label_value / num_label)
            num_append = factor - 1
            plus_one = np.random.random_sample() < ((max_label_value - (num_label * factor)) / num_label)
            if plus_one:
                num_append += 1
            more_idxs.extend([i] * num_append)
        train_idxs.extend(more_idxs)
        print('Num training examples balanced: ' + str(len(train_idxs)))
        print('Should be close to: ' + str(max_label_value * num_labels))

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
