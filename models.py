from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

# vectorizer = CountVectorizer(min_df=2, analyzer='char', ngram_range=(2,3))
vectorizer = CountVectorizer(min_df=2, analyzer='word', ngram_range=(1,1))

with open('adverbs.txt') as input_file:
    data = input_file.readlines()

labels = []
examples = []

# label_set = ['angrily', 'anxiously', 'bitterly', 'bowing', 'coldly', 'continuing', 'eagerly', 'excitedly', 'frightened', 'impatiently', 'indignantly', 'laughing', 'laughs', 'nervously', 'nodding', 'quickly', 'rising', 'sadly', 'severely', 'sharply', 'shouts', 'sighs', 'slowly', 'smiling', 'softly', 'startled', 'sternly', 'surprised', 'thoughtfully', 'weeping']
# min_label_count = 20

# label_set = ['angrily', 'bitterly', 'eagerly', 'impatiently', 'indignantly', 'laughing', 'nodding', 'rising', 'sadly', 'smiling', 'surprised']
# min_label_count = 30

label_set = ['angrili', 'laugh', 'nod', 'smile']
min_label_count = 200




label_counts = [1] * len(label_set)
print "Num of labels: ", len(label_set)
print "Random Guessing: ", 1.0/len(label_set)

for ex in data:
    label, prev_line, line, next_line = ex.split('\t')
    line = ' '.join([prev_line, line, next_line])
    # label, line = ex.split('\t')

    if label in label_set:
        label_idx = label_set.index(label)

        if label_counts[label_idx] <= min_label_count:
            label_counts[label_idx] = label_counts[label_idx] + 1
            labels.append(label_idx)
            examples.append(line)

    # if label == 'nodding':
    #     labels.append(1)
    #     examples.append(line)
    # elif label == 'aside':
    #     labels.append(0)
    #     examples.append(line)
    # elif label == 'smiling':
    #     labels.append(2)
    #     examples.append(line)

training_matrix = vectorizer.fit_transform(examples)

clf = LinearSVC()
cv_score = cross_val_score(clf, training_matrix, labels, cv=5).mean()
print("Cross validation score: {}".format(cv_score))
