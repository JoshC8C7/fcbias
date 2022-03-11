import pandas as pd
import csv
import json

d_dict = {}

with open('assertions.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        if row[1] == '/r/Synonym' or row[1] == '/r/SimilarTo':
            k = row[0].split(',')
            source = k[1].split('/a/wn/')[0].split('/n/wn/')[0].split('/v/wn/')[0].split('/r/wn/')[0]
            sink = k[2].split('/a/wn/')[0].split('/n/wn/')[0].split('/v/wn/')[0].split('/r/wn/')[0]

            if '/en/' in source and '/en/' in sink:
                if source not in d_dict:
                    d_dict[source] = []
                if sink not in d_dict:
                    d_dict[sink] = []
                if sink not in d_dict[source]:
                    d_dict[source].append(sink)
                    d_dict[sink].append(source)

with open("sim.json", "w",encoding='utf-8') as outfile:
    json.dump(d_dict, outfile)


#df.to_csv('conv99.csv',index=False)
print("ALL DONE")

"""df = pd.read_csv('assertions.csv', sep='\t')
print(df)"""
"""
df = pd.read_csv('conv.csv')
k = df[~df['wn'].str.contains('\+')]
k.to_csv('conv2.csv',index=False)
print("ALL DONE 2")"""
"""
df = pd.read_json('preps.json')
df2 = df['preposition']
print(df2)
df2.to_csv('prepsx.csv',index=False)"""
"""
def rearr(row):
    ent = row['wn30']
    return ent[1:]+'-'+ent[0]

def padd(row):
    ent = row['wn31']
    t = ent[1:]
    while len(t) < 8:
        t='0'+t
    return t+'-'+ent[0]

def stripz(row):
    ent = row['wn31']
    return ent[1:]

with open('wnconverter.json') as f:
    di = json.load(f)
    dfww = pd.DataFrame(di.items(),columns=['wn30','wn31'])


dfww['wn30'] = dfww.apply(rearr,axis=1)
dfww['wn31'] = dfww.apply(padd,axis=1)

dfcn = pd.read_csv('conv2.csv',names=['wn31','cn'])
dfcn['wn31'] = dfcn.apply(stripz,axis=1)
print(dfcn)
print(dfww)
merged = pd.merge(dfww, dfcn, on=['wn31'])
print(merged)
merged.to_csv('final-3waymap.csv',index=False)
dfx = merged[['wn30','cn']]
dfx.to_csv('final-2waymap.csv',index=False)
"""