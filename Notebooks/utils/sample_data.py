import random

def sample(train_bow, train_d2v, label):
    num = len(label)
    index_set = set(random.sample(range(num), int(num / 2)))
    
    l1_train_bow = []
    l1_train_d2v = []
    l1_label = []
    
    l2_train_bow = []
    l2_train_d2v = []
    l2_label = []
    
    for i in range(num):
        if i in index_set:
            l1_train_bow.append(train_bow[i])
            l1_train_d2v.append(train_d2v[i])
            l1_label.append(label[i])
        else:
            l2_train_bow.append(train_bow[i])
            l2_train_d2v.append(train_d2v[i])
            l2_label.append(label[i])
    
    return l1_train_bow, l1_train_d2v, l2_train_bow, l2_train_d2v, l1_label, l2_label
