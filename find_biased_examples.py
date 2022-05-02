import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
import json
import gc
import pickle
import pandas as pd
from spacy.pipeline import Lemmatizer
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token
from spacy.tokens.span import Span
from lemminflect import getInflection, getLemma
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as skcs
from negspacy.negation import Negex


def ws_tokenizer(txt):
    return Doc(nlp.vocab,txt.split(" "))

nlp = spacy.load('en_core_web_trf')
nlp.tokenizer = ws_tokenizer

def ratio_intersection(cl, ev):
    s1 = set(x.text for x in cl.ents)
    s2 = set(x.text for x in ev.ents)
    q1 = set(x.lemma for x in cl if (x.ent_iob_ == "O" and not x.is_stop))
    q2 = set(x.lemma for x in ev if (x.ent_iob_ == "O" and not x.is_stop))
    return (len(s1.intersection(s2)), len(q1.intersection(q2)), (len(s1.intersection(s2)) / max(1,len(q1.intersection(q2)))))

def find_entity_overweighting(dataset):
    ds = dataset.sample(n=100,random_state=100)
    claims = ds['claim']
    print(claims)
    listo = []

    cl_docs = nlp.pipe(ds['claim'],disable=["tok2vec", "tagger", "parser", "attribute_ruler", ])
    ev_docs = nlp.pipe(ds['evidence'],disable=["tok2vec", "tagger", "parser", "attribute_ruler", ])

    for x,y in zip(cl_docs,ev_docs):
        print(x.ents, y.ents,ratio_intersection(x,y))
        listo.append(ratio_intersection(x,y))
    ds['vals'] = listo
    ds.to_csv('metric_design_overweighting.csv')


    """Find ratio of intersection(hyp-entities, ev-entities) : intersection(hyp-NSNE, ev-NSNE)
    with NSNE = non-stop non-entity words
    potentially biased examples are ones that are basically just seeing entity overlap.
    """


    print("fish")
    return
def get_dataset_stream(dataset, split):

    """
    dataset one of [fever, vitc-all]
    split one of [test, train, dev]

    """
    if dataset=='fever':
        return pd.read_json('VitaminC/data/fever/' + split+'.jsonl',lines=True)

    elif dataset=='vitc-all':
        return pd.read_json('VitaminC/data/vitaminc/' + split+'.jsonl',lines=True)

    else:
        raise FileNotFoundError('No dataset by that name found')


def plot_sca(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    dfnei = df[df['labels'] == 'NOT ENOUGH INFO']
    dfsup = df[df['labels'] == 'SUPPORTS']
    dfref = df[df['labels'] == 'REFUTES']
    df['diff'] = df['entos'] - df['nsnes']

    ax = sns.stripplot(data=df,x='nsnes',y='entos',hue='labels',jitter=5,dodge=True)
    plt.xticks(rotation=90)

    plt.savefig('all.png')

    #ax = sns.countplot(data=df,x='diff',hue='labels')
    #plt.savefig('diff.png')

    #ax = sns.stripplot(data=dfref,x='nsnes',y='entos',hue='labels')

    #plt.savefig('ref.png')


    return


dataset = get_dataset_stream('fever','train')
#find_entity_overweighting(dataset)
ds2 = pd.read_csv('entow.csv')
plot_sca(ds2)