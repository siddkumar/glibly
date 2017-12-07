from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

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


for item in data:
    label, sentence = item.split('\t')
    if not label in counter or counter[label] < 10:
        continue

    orig_sentence = sentence.decode('utf-8')
    sentence = orig_sentence.lower()
    sentence = re.sub('[' + string.punctuation + ']' , '', sentence)

    stylometric_features = []
    scale = float(len(orig_sentence))
    for c in string.punctuation:
        stylometric_features.append(orig_sentence.count(c) / scale)

    labels.append(label)
    examples.append(stylometric_features)

print "Number of examples: ", len(examples)
# training_matrix = vectorizer.fit_transform(examples)
training_matrix = examples
print "Dilily Dilly"

clf = LinearSVC()
cv_score = cross_val_score(clf, training_matrix, labels, cv=5).mean()
print("Cross validation score: {}".format(cv_score))
