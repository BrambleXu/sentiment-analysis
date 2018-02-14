from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import bsr_matrix
import numpy as np
import pandas as pd
from sklearn.svm import SVC

import sys
sys.path.insert(0, '..')
from utils.feature_select import select_feature
from utils.TextPreprocess import review_to_words
from utils.sample_data import sample


'''
Training Data
'''
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0,
                         delimiter='\t', quoting=3, error_bad_lines=False)
num_reviews = train["review"].size

print("Cleaning and parsing the training set movie reviews...")
clean_train_reviews = []
for i in range(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))

'''
Test Data
'''
test = pd.read_csv("../../data/testData.tsv", header = 0, delimiter = "\t", quoting = 3)

num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...")
for i in range(0, num_reviews):
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)



train_data_features_d2v = np.loadtxt('../../data/train_feature_d2v.txt')
test_data_features_d2v = np.loadtxt('../../data/test_feature_d2v.txt')

lab_fea = select_feature('../../data/feature_chi.txt', 1000)['1']

result = [0.0 for i in range(num_reviews)]


max_iter = 5
for epoch in range(max_iter):
    print("epoch: " + str(epoch))
    l1_train_bow, l1_train_d2v, l2_train_bow, l2_train_d2v, l1_label, l2_label = sample(clean_train_reviews,
                                                                                        train_data_features_d2v,
                                                                                        train["sentiment"].values)

    print("training bow ...")
    vectorizer_bow = TfidfVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     vocabulary=lab_fea,
                                     max_features=19000)

    l1_train_features_bow = vectorizer_bow.fit_transform(l1_train_bow)
    l1_train_features_bow = bsr_matrix(l1_train_features_bow)

    l1_lr_bow = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,
                                   class_weight=None, random_state=None)
    l1_lr_bow = l1_lr_bow.fit(l1_train_features_bow, l1_label)

    l2_test_features_bow = vectorizer_bow.transform(l2_train_bow)
    l2_test_features_bow = bsr_matrix(l2_test_features_bow)

    l2_result_bow = l1_lr_bow.predict_proba(l2_test_features_bow)[:, 1]

    print("train doc2vec ...")
    l1_train_features_d2v = bsr_matrix(l1_train_d2v)
    l2_test_features_d2v = bsr_matrix(l2_train_d2v)

    l1_svm_d2v = SVC(C = 1.0, kernel='rbf', gamma = 'auto', probability=True)
    l1_svm_d2v = l1_svm_d2v.fit(l1_train_features_d2v, l1_label)

    l2_result_d2v = l1_svm_d2v.predict_proba(l2_test_features_d2v)[:, 1]

    print("train ensemble ...")
    train_data_features_ens = []

    for i in range(len(l2_result_bow)):
        vector = []
        vector.append(l2_result_bow[i])
        vector.append(l2_result_d2v[i])

        train_data_features_ens.append(vector)

    lr_ens = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,
                                class_weight=None, random_state=None)
    lr_ens = lr_ens.fit(train_data_features_ens, l2_label)

    print("final predict ...")
    train_bow = vectorizer_bow.fit_transform(clean_train_reviews)
    train_bow = bsr_matrix(train_bow)

    test_bow = vectorizer_bow.transform(clean_test_reviews)
    test_bow = bsr_matrix(test_bow)

    train_d2v = bsr_matrix(train_data_features_d2v)
    test_d2v = bsr_matrix(test_data_features_d2v)

    lr_bow = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,
                                class_weight=None, random_state=None)
    lr_bow = lr_bow.fit(train_bow, list(train["sentiment"]))

    svm_d2v = SVC(C = 1.0, kernel='rbf', gamma = 'auto', probability=True)
    svm_d2v = svm_d2v.fit(train_d2v, train["sentiment"].values)

    result_bow = lr_bow.predict_proba(test_bow)[:, 1]
    result_d2v = svm_d2v.predict_proba(test_d2v)[:, 1]

    test_data_features_ens = []

    for i in range(len(result_bow)):
        vector = []
        vector.append(result_bow[i])
        vector.append(result_d2v[i])

        test_data_features_ens.append(vector)

    result_test_ens = lr_ens.predict_proba(test_data_features_ens)[:, 1]

    for i in range(num_reviews):
        result[i] += result_test_ens[i]

for i in range(num_reviews):
    result[i] /= max_iter

result = np.array(result)
result_bool = result >= 0.5
combine = pd.DataFrame(data={'id': test['id'],
                             'sentiment': result_bool * 1})

print("output...")
combine.to_csv('../../result/ensemble.csv', index=False, quoting=3)