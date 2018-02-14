"""
这个文件训练两个deep模型，dbow和dm，段落向量为100维
最后的分类器选用了Logistic Regression
"""


# gensim modules
from gensim.models import Doc2Vec

# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression

# random
from random import shuffle

# preprocess packages
import pandas as pd
import sys
sys.path.insert(0, '..')
from utils.TextPreprocess import review_to_words, tag_reviews


'''
Training Data
'''
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter='\t',
                    quoting=3, error_bad_lines=False)
num_reviews = train["review"].size

print("Cleaning and parsing the training set movie reviews...")
clean_train_reviews = []
for i in range(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))

'''
Test Data
'''
test = pd.read_csv("../../data/testData.tsv", header = 0, delimiter = "\t",
                   quoting = 3)
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...")
for i in range(0, num_reviews):
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)


# Unlabeled Train Data
unlabeled_reviews = pd.read_csv("../../data/unlabeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
num_reviews = len(unlabeled_reviews["review"])
clean_unlabeled_reviews = []

print("Cleaning and parsing the test set movie reviews...")
for i in range( 0, num_reviews):
    if( (i+1)%5000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(unlabeled_reviews["review"][i])
    clean_unlabeled_reviews.append(clean_review)

# tag all reviews
train_tagged = tag_reviews(clean_train_reviews, 'TRAIN')
test_tagged = tag_reviews(clean_test_reviews, 'TEST')
unlabeled_train_tagged = tag_reviews(clean_unlabeled_reviews, 'UNTRAIN')


# model construction
model_dm = Doc2Vec(min_count=1, window=10, size=100, sample=1e-3, negative=5, dm=1, workers=3)
model_dbow = Doc2Vec(min_count=1, window=10, size=100, sample=1e-3, negative=5, dm=0, workers=3)

# build vocabulary
all_tagged = []
tag_objects = [train_tagged, test_tagged, unlabeled_train_tagged]
for tag_object in tag_objects:
    for tag in tag_object:
        all_tagged.append(tag)

model_dm.build_vocab(all_tagged)
model_dbow.build_vocab(all_tagged)

# train two model
train_tagged2 = []
tag_objects = [train_tagged, unlabeled_train_tagged]
for tag_object in tag_objects:
    for tag in tag_object:
        train_tagged2.append(tag)

for i in range(10):
    shuffle(train_tagged2)
    model_dm.train(train_tagged2, total_examples=len(train_tagged2), epochs=1, start_alpha=0.025, end_alpha=0.025)

for i in range(10):
    shuffle(train_tagged2)
    model_dbow.train(train_tagged2, total_examples=len(train_tagged2), epochs=1, start_alpha=0.025, end_alpha=0.025)


# get the vector of train and test data
train_array_dm = []
for i in range(len(train_tagged)):
    tag = train_tagged[i].tags[0]
    train_array_dm.append(model_dm.docvecs[tag])

train_array_dbow = []
for i in range(len(train_tagged)):
    tag = train_tagged[i].tags[0]
    train_array_dbow.append(model_dbow.docvecs[tag])

train_target = train['sentiment'].values

test_array_dm = []
for i in range(len(test_tagged)):
    test_array_dm.append(model_dm.infer_vector(test_tagged[i].words))

test_array_dbow = []
for i in range(len(test_tagged)):
    test_array_dbow.append(model_dbow.infer_vector(test_tagged[i].words))

# classification model
lr_dm = LogisticRegression()
lr_dbow = LogisticRegression()

# train
lr_dm.fit(train_array_dm, train_target)
lr_dbow.fit(train_array_dbow, train_target)

# predict
result_dm = lr_dm.predict(test_array_dm)
result_dbow = lr_dbow.predict(test_array_dbow)

# output
print("output...")
output_dm = pd.DataFrame(data={'id': test['id'], 'sentiment': result_dm})
output_dm.to_csv('../../result/doc2vec_dm.csv', index=False, quoting=3)

output_dbow = pd.DataFrame(data={'id': test['id'], 'sentiment': result_dbow})
output_dbow.to_csv('../../result/doc2vec_dbow.csv', index=False, quoting=3)
