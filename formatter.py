import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def id_morph(row):
    if row['id'][0] == 'm':
        return row['id'][2:]
    else:
        return row['id']

def lab_morph(row):
    lab = row['label']
    if lab == "SUPPORTS":
        return "REFUTES"
    elif lab == "NOT ENOUGH INFO":
        return "NOT ENOUGH INFO"
    elif lab == "REFUTES":
        return "SUPPORTS"
    else:
        raise Exception
"""
#df = pd.read_csv('adversaries/fever_dev_neg_overlap_semiauto_2000.csv')

def stripnc(row):
    cl = row['new_claims']
    cl2 = cl.split(". ")[0]
    cl3 = cl2.replace("<unk>","")
    return cl3

df2 = pd.read_csv('done_neg_overlap_fever_dev_auto_UG.csv')
print(df2)
df2['new'] = df2.apply(stripnc,axis=1)
print(df2)
df4 = df2.drop(['new_claims','FEVER_id','revision_type','page','wiki_revision_id','unique_id','case_id'],axis=1)
df3 = df2.drop(['claim','new_claims'],axis=1)
df3=df3.rename({'new':'claim'},axis=1)
#print(df3)
#df3.to_json('overweighting.jsonl',orient='records',lines=True)
df5 = df4.sample(random_state=99,n=150)
print(df5)
df5.to_csv('overwewighting_2nd_sample.csv',index=False)"""


df = pd.read_csv('done_neg_overlap_fever_dev_auto_UG.csv')
df['id2'] = df['id2'].astype(str)

#df3= df.rename({'new_claims':'claim'},axis=1)
#df['id2'] = df.apply(id_morph,axis=1)
#df.to_json('adversaries/fever-dev-tfidf_overlap_new.jsonl',orient='records',lines=True)
df_big = pd.read_json('VitaminC/data/fever/dev.jsonl',lines=True)
df_big['id2'] = df_big['id'].astype(str)
df3 =df.merge(df_big,how='left',on='id2')
df3 = df3.drop(['claim','id2','new_claims_antonym','new_claims_negation'],axis=1)
df3= df3.rename({'new_claims_both':'claim'},axis=1)
print(df3)
#df3 =df3.drop(['new_label'],axis=1)
#print(df)
df3.to_json('adversaries/neg_overlap_fever_dev_fullauto_ungrammatical.jsonl',orient='records',lines=True)
df9 = df3.sample(random_state=99,n=100)
df9.to_csv('neg_overlap_fever_dev_fullauto_sample.csv',index=False)