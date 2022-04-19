import sys
import gzip
from lemminflect import getLemma
import pandas as pd
from nltk.tag import pos_tag

def get_ppdb(ppdbfile) :
    ppdb = set()
    with gzip.open(ppdbfile,'rt') as z :
        for line in z:
            text = line.split(' ||| ')
            if len(text) == 6 and text[5].strip() == 'Equivalence':
                _, x, y, _ = line.strip().split(' ||| ', 3)
                try:
                    x_pos = pos_tag([x],tagset='universal')[0][1]
                    y_pos = pos_tag([y],tagset='universal')[0][1]
                    if x_pos == 'ADP' or y_pos == 'ADP':
                        raise Exception
                    p2 = (getLemma(x,x_pos)[0],getLemma(y,y_pos)[0])
                    if p2[0] == p2[1]: continue
                    ppdb.add(p2)

                except:
                    print("tuple access failed")
                    ppdb.add((x,y))

    return ppdb


sys.stderr.write('loading PPDB.\n')
ppdb = get_ppdb('ppdb-2.0-xxl-all.gz')

sys.stderr.write('done loading PPDB.\n')
df = pd.DataFrame(list(ppdb),columns=['source','sink'])


sys.stderr.write('writing antonyms.\n')
df.to_csv('ppdb_syn_map.csv',index=False)

sys.stderr.write('done writing antonyms.\n')