import json
import os
from pathlib import Path

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data = []
data_labels = []

# pre_data = open('train_data_pos_neg.bin', 'rw+')
# pre_data_labels = open('train_data_labels_pos_neg.bin', 'rw+')
my_file = Path("train_data_pos_neg.bin")
count = 0
limit = 2000
if not my_file.is_file():
    for dirName, subdirList, fileList in os.walk("./training_data/positiveReviews/"):
        # print('Found directory: %s' % dirName)
        for fname in fileList:
            count += 1
            if count > limit:
                break
            f = open(dirName + fname, "r")
            data.append(f.readlines()[0])
            data_labels.append('pos')
            print(fname + ' pos')
    count = 0
    for dirName, subdirList, fileList in os.walk("./training_data/negativeReviews/"):
        # print('Found directory: %s' % dirName)
        count += 1
        if count > limit:
            break
        for fname in fileList:
            f = open(dirName + fname, "r")
            data.append(f.readlines()[0])
            data_labels.append('neg')
            print(fname + ' neg')
    pre_data = open('train_data_pos_neg.bin', 'w')
    pre_data_labels = open('train_data_labels_pos_neg.bin', 'w')

    json.dump(data, pre_data)
    pre_data.close()
    json.dump(data_labels, pre_data_labels)

    pre_data_labels.close()
else:
    pre_data = open('train_data_pos_neg.bin', 'r')
    pre_data_labels = open('train_data_labels_pos_neg.bin', 'r')

    data = json.load(pre_data)
    data_labels = json.load(pre_data_labels)
    # print(data, data_labels)

#
# with open("./neg_tweets.txt") as f:
#     for i in f:
#         data.append(i)
#         data_labels.append('neg')
print('Using Logisitic Regression\n==========================================\n')

vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)

features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()  # for easy usage
print('================================\nNumber of unique words: ' + str(len(features_nd.tolist()[0])) + '\n')
# print(features_nd.dtype)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    data_labels,
    train_size=0.60)

from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron

# clean before after
# confusion
# compare logistic
log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

# print(y_pred, X_test, '\n')
import random

j = random.randint(0, len(X_test) - 2)
for i in range(j, j + 2):
    # print(y_pred[i])
    ind = features_nd.tolist().index(X_test[i].tolist())
    # print(data[ind].strip())

from sklearn.metrics import accuracy_score, confusion_matrix

print('\n=================================\nAccuracy of Logistic Regression: ' + str(accuracy_score(y_test, y_pred)))

print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
      '\n\n\n================================================\n\n\n\n\n\n')










#========================================================TEST 2====================================================


#
#
# print('Now without including stop words i.e cleaning')
#
#
# vectorizer = CountVectorizer(
#     analyzer='word',
#     lowercase=False,
#     stop_words='english'
# )
#
# features = vectorizer.fit_transform(
#     data
# )
# features_nd = features.toarray()  # for easy usage
# print('================================\nNumber of unique words: ' + str(len(features_nd.tolist()[0])) + '\n\n\n')
# # print(features_nd.dtype)
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     features_nd,
#     data_labels,
#     train_size=0.60)
#
# from sklearn.linear_model import LogisticRegression
#
# # clean before after
# # confusion
# # compare logistic
# log_model = LogisticRegression()
#
# log_model = log_model.fit(X=X_train, y=y_train)
# y_pred = log_model.predict(X_test)
#
# # print(y_pred, X_test, '\n')
# import random
#
# j = random.randint(0, len(X_test) - 2)
# for i in range(j, j + 2):
#     print(y_pred[i])
#     ind = features_nd.tolist().index(X_test[i].tolist())
#     print(data[ind].strip())
#
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# print('\n=================================\nAccuracy of Logistic Regression: ' + str(accuracy_score(y_test, y_pred)))
#
# print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
#       '\n\n\n================================================\n\n\n\n\n\n')




#========================================================TEST 2====================================================




print('Now with Perceptron')


vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
    stop_words='english'
)

features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()  # for easy usage
# print(features_nd.dtype)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    data_labels,
    train_size=0.60)


# clean before after
# confusion
# compare logistic
log_model = Perceptron()

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

# print(y_pred, X_test, '\n')
import random

j = random.randint(0, len(X_test) - 2)
for i in range(j, j + 2):
    # print(y_pred[i])
    ind = features_nd.tolist().index(X_test[i].tolist())
    # print(data[ind].strip())

from sklearn.metrics import accuracy_score, confusion_matrix

print('\n=================================\nAccuracy of Perceptron: ' + str(accuracy_score(y_test, y_pred)))

print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
      '\n\n\n================================================\n\n\n\n\n\n')










#============================================================================================================




#========================================================TEST 2====================================================




print('Now with Decision Tree classifier ')


vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)

features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()  # for easy usage
# print(features_nd.dtype)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    data_labels,
    train_size=0.60)


# clean before after
# confusion
# compare logistic
log_model = DecisionTreeClassifier(max_depth=10)

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

# print(y_pred, X_test, '\n')
import random

j = random.randint(0, len(X_test) - 2)
for i in range(j, j + 2):
    # print(y_pred[i])
    ind = features_nd.tolist().index(X_test[i].tolist())
    # print(data[ind].strip())

from sklearn.metrics import accuracy_score, confusion_matrix

print('\n=================================\nAccuracy of DecisionTreeClassifier with max_depth 10: ' + str(accuracy_score(y_test, y_pred)))

print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
      '\n\n\n================================================\n\n\n\n\n\n')









#============================================================================================================




#========================================================TEST 2====================================================




print('Now with Naive Bayes Gaussian')


vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)

features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()  # for easy usage
# print(features_nd.dtype)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    data_labels,
    train_size=0.60)


# clean before after
# confusion
# compare logistic
log_model = GaussianNB()

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

# print(y_pred, X_test, '\n')
import random

j = random.randint(0, len(X_test) - 2)
for i in range(j, j + 2):
    # print(y_pred[i])
    ind = features_nd.tolist().index(X_test[i].tolist())
    # print(data[ind].strip())

from sklearn.metrics import accuracy_score, confusion_matrix

print('\n=================================\nAccuracy of NaiveBayes: ' + str(accuracy_score(y_test, y_pred)))

print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
      '\n\n\n================================================\n\n\n\n\n\n')
# from sklearn.feature_extraction.text import CountVectorizer
#
#
#
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)
#
# count_vect.vocabulary_.get(u'algorithm')
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# print(X_train_tf.shape)
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)
#
# # from sklearn.naive_bayes import MultinomialNB
# #
# # clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# #
# # docs_new = ['God is love', 'OpenGL on the GPU is fast']
# # X_new_counts = count_vect.transform(docs_new)
# # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# #
# # predicted = clf.predict(X_new_tfidf)
# #
# # for doc, category in zip(docs_new, predicted):
# #     print('%r => %s' % (doc, twenty_train.target_names[category]))
#
# # from sklearn.pipeline import Pipeline
# # text_clf = Pipeline([('vect', CountVectorizer()),
# #                      ('tfidf', TfidfTransformer()),
# #                      ('clf', MultinomialNB()),
# # ])
# # text_clf.fit(twenty_train.data, twenty_train.target)
#
#
# # TESTING=============================================
#
#
#
# # import numpy as np
# #
# # twenty_test = fetch_20newsgroups(subset='test',
# #                                  categories=categories, shuffle=True, random_state=42)
# # docs_test = twenty_test.data
# # predicted = text_clf.predict(docs_test)
# # np.mean(predicted == twenty_test.target)
# #
# # from sklearn.linear_model import SGDClassifier
# #
# # text_clf = Pipeline([('vect', CountVectorizer()),
# #                      ('tfidf', TfidfTransformer()),
# #                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
# #                                            alpha=1e-3, random_state=42,
# #                                            max_iter=5, tol=None)),
# #                      ])
# # text_clf.fit(twenty_train.data, twenty_train.target)
# #
# # predicted = text_clf.predict(docs_test)
# # np.mean(predicted == twenty_test.target)