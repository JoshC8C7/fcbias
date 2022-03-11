"""
Utility to trim the 2waymap from wn to cn as to remove trailing /a/wn/...
for some reason concept net has some edges listed as this, despite them being identical
to the equivalent trimmed one.
"""
import pandas as pd

def trim(row):
    return row['source'].split('/wikt/')[0]


df = pd.read_csv('conv99.csv')
print(df)
df['source'] = df.apply(trim,axis=1)
df.to_csv('cn_antonym_mapxxx.csv',index=None)