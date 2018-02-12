from bs4 import BeautifulSoup
import pandas as pd

train_data = pd.read_csv("../../data/labeledTrainData.tsv",
                         header = 0, delimiter = "\t", quoting = 3, error_bad_lines=False)

bw_pos = open('train/data/full-train-pos.txt', 'w')
bw_neg = open('train/data/full-train-neg.txt ', 'w')

bw_train = open('train/data/full-train.txt', 'w')

num_reviews = train_data["review"].size
print("Cleaning and dumping the training set movie reviews...")
for i in range(0, num_reviews):
    text = BeautifulSoup(train_data["review"][i], 'lxml').get_text()

    text = text.strip('"')
    text = text.replace('\\', '')

    label = train_data["sentiment"][i]
    if label == 1:
        bw_pos.write(text + '\n')
    else:
        bw_neg.write(text + '\n')

    bw_train.write(text + '\n')

bw_pos.close()
bw_neg.close()
bw_train.close()

test_data = pd.read_csv("../../data/testData.tsv", header = 0,
                   delimiter = "\t", quoting = 3)

num_reviews = len(test_data["review"])
bw_test = open('full-test.txt', 'w')

print("Cleaning and dumping the testing set movie reviews...")
for i in range(0, num_reviews):
    text = BeautifulSoup(test_data["review"][i], 'lxml').get_text()

    text = text.strip('"')
    text = text.replace('\\', '')

    bw_test.write(text + '\n')
bw_test.close()



unlabel_data = pd.read_csv("../../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = len(test_data["review"])
bw_unlabel = open('full-unlabeled.txt', 'w')

print("Cleaning and dumping the unlabeled set movie reviews...")
for i in range(0, num_reviews):
    text = BeautifulSoup(unlabel_data["review"][i], 'lxml').get_text()
    text = text.strip('"')
    text = text.replace('\\', '')

    bw_unlabel.write(text + '\n')
bw_unlabel.close()
    
