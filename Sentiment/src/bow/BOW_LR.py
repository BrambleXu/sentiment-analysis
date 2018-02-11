'''
Created on 2014年12月16日

@author: GongYu

Modified on 2018年2月10日

@author：BrambleXu
'''
import sys
sys.path.insert(0, '..')
from utils.feature_select import select_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import bsr_matrix
import numpy as np

class BagOfWords(object):
    
    def __init__(self, vocab = False, tfidf = False, max_feature = 1000):
        lab_fea = None
        if(vocab == True):
            print("select features...")
            lab_fea = select_feature('../../data/feature_chi.txt', max_feature)["1"]
        
        self.vectorizer = None
        if(tfidf == True):
            self.vectorizer = TfidfVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 vocabulary = lab_fea,
                                 max_features = max_feature)
        else:
            self.vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 vocabulary = lab_fea,
                                 max_features = max_feature)
        self.lr = None
        
    def train_lr(self, train_data, lab_data, C = 1.0):
        train_data_features = self.vectorizer.fit_transform(train_data)
        train_data_features = bsr_matrix(train_data_features)
        print(train_data_features.shape)
        
        print("Training the logistic regression...")
        self.lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, fit_intercept=True,
                                     intercept_scaling=1.0, class_weight=None, random_state=None)
        self.lr = self.lr.fit(train_data_features, lab_data)
        
    def test_lr(self, test_data):
        test_data_features = self.vectorizer.transform(test_data)
        test_data_features = bsr_matrix(test_data_features)
    
        result = self.lr.predict(test_data_features)
        return result
    
    def validate_lr(self, train_data, lab_data, C = 1.0):
        train_data_features = self.vectorizer.fit_transform(train_data)
        train_data_features = bsr_matrix(train_data_features)
        lab_data = np.array(lab_data)
        
        print("start k-fold validate...")
        lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
        cv = np.mean(cross_val_score(lr, train_data_features, lab_data, cv=10, scoring='roc_auc'))
        return cv
    