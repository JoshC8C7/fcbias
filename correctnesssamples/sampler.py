import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
seed = 99

df = pd.read_json('adversaries/ungram/negation_overlap_fever_test_ungrammatical.jsonl',lines=True)
print(df)
df2 = (df.sample(random_state=seed,n=100))
df2.to_csv('neg_overlap_fever_test_sample.csv',index=False)