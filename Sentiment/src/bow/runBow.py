'''
Created on 2014年12月16日

@author: GongYu

Modified on 2018年2月10日

@author：BrambleXu
'''
import pandas as pd
import sys
sys.path.insert(0, '..')
from utils.TextPreprocess import review_to_words

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

'''
Train and Test
'''
import BOW_LR

bow = BOW_LR.BagOfWords(vocab = True, tfidf = True, max_feature = 19000)
bow.train_lr(clean_train_reviews, list(train["sentiment"]), C = 1)
result = bow.test_lr(clean_test_reviews)
print(result)

print("output...")
output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
output.to_csv('../../result/BOW_chi_tfidf.csv', index=False, quoting=3)