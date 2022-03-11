import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

"""
big = pd.read_json('VitaminC/data/fever/test.jsonl',lines=True)
small = pd.read_csv('test/p1.csv')

big['id_local'] = big.index
out = big.merge(small,on='id_local',how='inner')
out2 = out[['id','label','new_claims']]
print(out2)
out2.to_csv('test/negwritten.csv')"""

multi = pd.read_json('VitaminC/data/fever/dev.jsonl',lines=True)
news = pd.read_csv('dev/negwritten.csv')
print(multi)
print(news)
out3 = multi.merge(news,on=['id','label'])
out3.to_csv('dev/final_negwritten.csv')