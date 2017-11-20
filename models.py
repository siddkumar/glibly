from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

vectorizer = CountVectorizer(min_df=2, analyzer='char', ngram_range=(2, 2))

with open('adverbs.txt') as input_file:
    data = input_file.readlines()

labels = []
examples = []

for ex in data:
    label, line = ex.split('\t')
    if label == 'angrily':
        labels.append(1)
        examples.append(line)
    elif label == 'smiling':
        labels.append(0)
        examples.append(line)
    elif label == 'laughing':
        labels.append(2)
        examples.append(line)

training_matrix = vectorizer.fit_transform(examples)

clf = LinearSVC()
cv_score = cross_val_score(clf, training_matrix, labels, cv=5).mean()
print("Cross validation score: {}".format(cv_score))
