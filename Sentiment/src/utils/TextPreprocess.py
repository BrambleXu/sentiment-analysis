import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join(meaningful_words))


def review_to_sentences(raw_review):
    #english_punctuations = [u',', u'.', u':', u';', u'?', u'(', u')', u'[', u']', u'&', u'!', u'*', u'@', u'#', u'$', u'%']
    review_text = BeautifulSoup(raw_review.strip()).get_text()
    
    sens = nltk.sent_tokenize(review_text)
    
    sentences = []
    for sent in sens:
        letters_only = re.sub("[^a-zA-Z0-9]", " ", sent) 
        words = letters_only.lower().split()
        sentences.append(words)
    return sentences

def review_to_ngram(raw_review, grams = [1, 2, 3]):
    review_text = BeautifulSoup(raw_review).get_text()    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    for gram in grams:
        tokens = []
        for i in range(len(meaningful_words) - gram + 1):
            tokens.append("_*_".join(meaningful_words[i:i+gram]))
    result = " ".join(tokens)
    #print result
    return result

def tag_reviews(reviews, prefix):
    tagged = []
    for i, review in enumerate(reviews):
        tagged.append(TaggedDocument(words=review.split(), tags=[prefix + '_%s' % i]))
    return tagged