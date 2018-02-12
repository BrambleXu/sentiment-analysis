import pandas as pd

test_data = pd.read_csv("data\\testData.tsv", header = 0, delimiter = "\t", quoting = 3)
num_reviews = len(test_data["id"])

br_result = open('result\\PARAGRAPH-TEST', 'r')
result_list = []

for line in br_result:
    line = line.strip()
    strs = line.split()
    result_list.append(float(strs[1]))
br_result.close()

out = open("result\\doc2vec.csv", 'w')
out.write("\"id\"" + "," + "\"sentiment\"")
out.write("\n")
for i in xrange(num_reviews):
    out.write(test_data["id"][i] + "," + str(result_list[i]) + "\n")
out.close()