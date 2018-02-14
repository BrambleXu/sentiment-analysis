import pandas as pd

doc2vec_result = pd.read_csv('../../result/doc2vec_lr100_prob.csv', header=0)
bow_result = pd.read_csv('../../result/bow_lr_prob.csv', header=0)


combine = pd.DataFrame(data={'id': bow_result['id'],
                             'sentiment': (bow_result['sentiment']+doc2vec_result['sentiment']) / 2})

combine['sentiment'] = combine['sentiment'] >= 0.5
combine['sentiment'] = combine['sentiment'].astype('int')

print("output...")
combine.to_csv('../Sentiment/result/combine.csv', index=False, quoting=3)
