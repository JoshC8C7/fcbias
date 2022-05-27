import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
seed = 99

df = pd.read_json('../adversaries/fever_dev_neg_overlap_semiauto_2000_antandneg.jsonl',lines=True)
print(df)
df2 = (df.sample(random_state=seed,n=100))
print(df2)
df2.to_csv('../tmp/indices.csv',index=True)