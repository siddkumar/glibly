from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

vectorizer = CountVectorizer(min_df=2, analyzer='char', ngram_range=(2, 3))
# vectorizer = CountVectorizer(min_df=2, analyzer='word', ngram_range=(1,1))


def svm(train_examples, train_labels, test_examples, test_labels, num_labels, embedding_dim):
    training_matrix = vectorizer.fit_transform(train_examples)
    test_matrix = vectorizer.transform(test_examples)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(training_matrix, train_labels)
    preds = clf.predict(test_matrix)
    probs = clf.predict_proba(test_matrix)
    print confusion_matrix(preds, test_labels)
    print accuracy_score(preds, test_labels)

    correct_r_3 = 0.0
    correct_r_5 = 0.0
    for ex_idx in xrange(len(test_examples)):
        lps = [(i, probs[ex_idx][i]) for i in xrange(len(probs[ex_idx]))]
        slps = sorted(lps, key=lambda tup: tup[1], reverse=True)
        top_3 = [slps[t][0] for t in range(3)]
        top_5 = [slps[t][0] for t in range(5)]

        if test_labels[ex_idx] in top_3:
            correct_r_3 += 1.0
            correct_r_5 += 1.0
        elif test_labels[ex_idx] in top_5:
            correct_r_5 += 1.0

    print "Dev Accuracy, Recall 3: " + str(correct_r_3/len(test_labels))
    print "Dev Accuracy, Recall 5: " + str(correct_r_5/len(test_labels))
