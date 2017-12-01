from corpus_stats import get_counter

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize

import numpy as np
import tensorflow as tf

filename = 'adverbs.txt'
max_sequence_len = 60
embedding_dim = 300
percent_training = 0.9
# WORD_VEC_FILE = '/Users/skumar/Documents/proj/nlp/GoogleNews-vectors-negative300.bin'
WORD_VEC_FILE = '/Users/skumar/Documents/proj/nlp/trained_vectors.bin'

with open(filename) as input_file:
    counter = get_counter(filename)
    data = input_file.readlines()

min_cutoff = 10
filtered_counter = {key:value for key,value in counter.iteritems() if value > min_cutoff}

label_set = filtered_counter.keys()
num_examples = len(data)
num_labels = len(label_set)
print('Number of labels: ' + str(num_labels))

examples = []
labels = np.zeros(num_examples)
seq_lens = np.zeros(num_examples)

wvModel = KeyedVectors.load_word2vec_format(WORD_VEC_FILE, binary=True)

s = 'Dilly dilly'

# Do the data
for i, item in enumerate(data):
    if i % 5000 == 0:
        print(s[i / 5000])
    label, sentence = item.split('\t')
    if label not in label_set:
        num_examples -= 1
        continue

    tokens = word_tokenize(sentence.decode('utf-8'))
    sentence_vectors = []
    for tok in tokens:
        if tok in wvModel:
            sentence_vectors.append(wvModel[tok])
        else:
            sentence_vectors.append(np.zeros(embedding_dim))

    examples.append(sentence_vectors)
    labels[i] = label_set.index(label)
    seq_lens[i] = len(tokens)

# Partition the data
shuffled_idxs = range(num_examples)
np.random.shuffle(shuffled_idxs)
num_training = int(percent_training*num_examples)

train_idxs = shuffled_idxs[:num_training]
train_examples = [examples[i] for i in train_idxs]
train_labels = [labels[i] for i in train_idxs]

test_idxs = shuffled_idxs[num_training:]
test_examples = [examples[i] for i in test_idxs]
test_labels = [labels[i] for i in test_idxs]

# Let the fun begin
print('Dilly dilly')

inputs = tf.placeholder(tf.float32, embedding_dim)
w = tf.get_variable('w', [num_labels, embedding_dim], initializer=tf.contrib.layers.xavier_initializer(seed=0))
z = tf.tensordot(w, inputs, 1)
probs = tf.nn.softmax(z)
one_best = tf.argmax(probs)
label = tf.placeholder(tf.int32, 1)
label_onehot = tf.reshape(tf.one_hot(label, num_labels), shape=[num_labels])
loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


global_step = tf.contrib.framework.get_or_create_global_step()
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
num_epochs = 10

with tf.Session() as sess:
    # Generally want to determinize training as much as possible
    tf.set_random_seed(0)
    # Initialize variables
    sess.run(init)
    step_idx = 0
    for i in range(0, num_epochs):
        loss_this_iter = 0
        # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
        for ex_idx in xrange(0, len(train_examples)):
            # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
            # causes training to happen
            [_, loss_this_instance] = sess.run([train_op, loss], feed_dict = {inputs: np.mean(train_examples[ex_idx], axis=0),
                                                                              label: np.array([train_labels[ex_idx]])})
            loss_this_iter += loss_this_instance
        print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

