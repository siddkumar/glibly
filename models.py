from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score

from corpus_stats import get_counter
import re
import string

vectorizer = CountVectorizer(min_df=2, analyzer='char', ngram_range=(2,3))
# vectorizer = CountVectorizer(min_df=2, analyzer='word', ngram_range=(1,1))

labels = []
examples = []

filename = 'adverbs.txt'
with open(filename) as input_file:
    counter = get_counter(filename)
    data = input_file.readlines()


clusters = {}
for i, l in enumerate(open('clusters.txt')):
    for w in l.split():
        clusters[w] = i

for item in data:
    label, sentence = item.split('\t')
    if not label in counter or counter[label] < 10 or label not in clusters:
        continue

    orig_sentence = sentence.decode('utf-8')
    sentence = orig_sentence.lower()
    sentence = re.sub('[' + string.punctuation + ']' , '', sentence)

    stylometric_features = []
    scale = float(len(orig_sentence))
    for c in string.punctuation:
        stylometric_features.append(orig_sentence.count(c) / scale)

    labels.append(clusters[label])
    examples.append(stylometric_features)

print "Number of examples: ", len(examples)
# training_matrix = vectorizer.fit_transform(examples)
training_matrix = examples
print "Dilily Dilly"

clf = LinearSVC()
cv_score = cross_val_score(clf, training_matrix, labels, cv=5).mean()


test_set_size = 1000
clf.fit(training_matrix[test_set_size:], labels[test_set_size:])
preds =  clf.predict(training_matrix[:test_set_size])
print confusion_matrix(preds, labels[:test_set_size])
print accuracy_score(preds, labels[:test_set_size])

count = [0] * 19
for i, v in clusters.items():
    count[v] += counter[i]
print count
print max(count) / sum(count)

print("Cross validation score: {}".format(cv_score))
