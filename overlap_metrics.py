from textacy.similarity import jaccard
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import textacy
import spacy
import pandas as pd
#nlp = spacy.load("en_core_web_lg")

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity as skcs
from sentence_transformers import SentenceTransformer, util
transf = SentenceTransformer('all-MiniLM-L6-v2')

"""Testing 'overlap' metrics:

x1. tf-idf 
x2. spaCy
x3. sBERT - https://www.sbert.net/docs/usage/semantic_textual_similarity.html
x5. Jaccard on non-stop-words (Via spaCy)
x6. MiniLM - https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1 / https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
x7. Jaccard - https://textacy.readthedocs.io/en/0.11.0/api_reference/similarity.html#textacy.similarity.tokens.jaccard

"""


#sktfid = TfidfVectorizer()

# Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings



def minilm(q1, q2):
    # Sentences we want sentence embeddings for
    query = q1
    docs = [q2]
    # Encode query and docs
    query_emb = encode(query)
    doc_emb = encode(docs)

    # Compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

    # Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    # Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    # Output passages & scores
    for doc, score in doc_score_pairs:
        return score

def tfidf(q1, q2):
    samps = [q1,q2]
    tfm = sktfid.fit_transform(samps)
    cosi = skcs(tfm,tfm)
    return(cosi[0][1])


def bow_overlap(d1, d2):
    s1 = set()
    s2 = set()
    for tok in d1:
        if not tok.is_stop:
            s1.add(tok.lemma_)
    for tok in d2:
        if not tok.is_stop:
            s2.add(tok.lemma_)
    return len(s1.intersection(s2))/max(len(s1),len(s2))

def sbert(s1,s2):
    sentences1 = [s1]
    sentences2 = [s2]
    embeddings1 = transf.encode(sentences1)
    embeddings2 = transf.encode(sentences2)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return float(cosine_scores[0][0])

ents = []

df = pd.read_json('VitaminC/data/fever/dev.jsonl',lines=True).sample(50,random_state=99)
"""
evs, claims, labs, tfids, spacys, sberts, jaccardstops, jaccards, minilms =[],[],[],[],[],[],[],[],[]

for i in df.iterrows():
    print("//////////////////////////")
    a = i[1]['evidence']
    b = i[1]['claim']
    aa = nlp(a)
    ab = nlp(b)
    evs.append(a)
    claims.append(b)
    #labs.append(i[1]['label'])
    print(a,"\n",b,"\n", )
    #tfids.append(tfidf(a,b))
    #spacys.append(aa.similarity(ab))
    sberts.append(sbert(a,b))
    jaccardstops.append(bow_overlap(aa,ab))
    jaccards.append(jaccard(a,b))
    minilms.append(minilm(a,b))

df2 = pd.DataFrame({'label':labs, 'tfids':tfids, 'spacy':spacys, 'sbert':sberts, 'jaccardstops':jaccardstops, 'jaccards':jaccards, 'minilms':minilms,'evs':evs,'claims':claims})
print(df2)
df2.to_csv('overlap_trials')"""
import timeit

t1 = '''
df = pd.read_json('VitaminC/data/fever/dev.jsonl',lines=True).sample(50,random_state=99)

for i in df.iterrows():
    minilm(i[1]['evidence'],i[1]['claim'])
'''

t2 = '''
df = pd.read_json('VitaminC/data/fever/dev.jsonl',lines=True).sample(50,random_state=99)

for i in df.iterrows():
    sbert(i[1]['evidence'],i[1]['claim'])
'''
print(timeit.timeit(stmt=t1))
