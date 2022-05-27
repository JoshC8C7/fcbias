import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
import json
import gc
import pickle
import pandas as pd
from spacy.pipeline import Lemmatizer
from spacy.tokens import Doc, Token
from spacy.tokens.span import Span
from lemminflect import getInflection, getLemma
from PyDictionary import PyDictionary
dictionary=PyDictionary()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as skcs
sktfid = TfidfVectorizer()

transf = SentenceTransformer('all-MiniLM-L6-v2')
nlp_min = English()

NEG_GO = True
if NEG_GO:
    from ewiser.spacy.disambiguate import Disambiguator
    # noinspection PyUnresolvedReferences
    from negspacy.negation import Negex

    with open('sim.json') as f:
        simconv = json.load(f)
    preplookup = pd.read_csv('preps.csv', header=None, index_col=0, squeeze=True).to_dict()
    wnconvert = pd.read_csv('final-wn2cn.csv', header=None, index_col=0, squeeze=True).to_dict()
    antsearch = pd.read_csv('cn_antonym_map.csv', header=None, index_col=0, squeeze=True).to_dict()
    ppdb_s = pd.read_csv('ppdb_ant_map.csv', header=None,index_col=0,squeeze=True).to_dict()

#Available models:
"""
From VitC:
Albert-base-vitaminc - huggingface/tals/albert-base-vitaminc
Albert-base-vitaminc-fever -  huggingface/tals/albert-base-vitaminc-fever *
Albert-base with fever only -  ./models/fever

*all sup and ref fever, randomly sampled NEI selection, see paper.

"""
"""
Unavailable / todo:

Fever Baselines (Yoneda implementation):
https://github.com/takuma-yoneda/fever-baselines
A: MLP-vitc
B: MLP-vitc-fever
C: MLP-fever

A: DA-vitc
B: DA-vitc-fever
C: DA-fever

Yoneda/UCL_MR system:
https://github.com/takuma-yoneda/fever-uclmr-system

Papelo-Transformer: https://aclanthology.org/W18-5517/
https://github.com/cdmalon/finetune-transformer-lm
    A: With vitaminc
    B: Vitaminc + fever (50/50)
    C: Fever only


"""

"""
Available adversarial datasets:

negation_mismatch
antonym
overlap
num_mismatch
named_ent_overweight

Pre-existing datasets:
Vitc
vitc_real
vitc_synthetic
fever

todo:
fever_adversarial
fever_symmetric
fever_triggers


"""




class datasetR():
    handle = None

    def __init__(self,path):
        print("Opening file handle")
        self.handle = open(path,'rb')

    def __del__(self):
        print("Closing file handle")
        self.handle.close()

    def __next__(self):
        return json.loads(next(self.handle))

    def __iter__(self):
        return self


#N.B To train albert, see run_train.sh

def predict_albert(data_dir, model):
    #Use {model} to predict data from data_dir, yields summary metrics.
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

"""
General rules:
1. No changing the evidence unless its other actual wikipedia evidence! 
Otherwise the training examples would always be impossible to get correct under a real test.

2. Derived hyps/evidence must be grammatical/sensical.
"""

#Negations be here
if NEG_GO:


    def generate_neg_mismatch(datasetx):
        dataset = datasetx.sample(100)
        LABEL_DICT = {"REFUTES": -1, "SUPPORTS": 1, "NOT ENOUGH INFO": 0}
        #print(dataset)
        #wsd = Disambiguator('ewiser/ewiser.semcor+wngt.pt')
        #wsd.to('cpu')
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("negex")
        #wsd.enable(nlp, 'wsd')
        """
        Where "-" indicates negation-word laden, and "+" indicates negation-word free:
        Adversaries are:
        (1) Hyp +, Text +, REF
        (2) Hyp -, Text -, REF
        (3) Hyp -, Text +, SUP
        (4) Hyp +, Text -, SUP
    
        Conversions:
        1. Positives:
        a. Double negate verbs (via negation word + antonym)
        b. Negate the noun:
            e.g. 'Trollhunters was produced by an animation company'
            -> 'Trollhunters was not produced by a dance company'.
        c. Negate the preposition:
            e.g. 'Steve Wozniak was born after the Apple II came out'
            -> 'Steve Wozniak was not born before the Apple II came out'.
    
        2. Negatives (¬2) and (¬3) - done manually
    
        """



        """"
        Do negation, get WSD sense, find antonym, replace object.
        """

        """
            Patterns:
    
            is -> is not
            are -> are not
            may -> may not
            could -> could not
            should -> should not
            was -> was not
            were -> were not
            have -> have not
    
            has (aux) -> has not
            [X]s (tag = VVZ) verb -> does not [X]
            [X]ed (tag = VVN) verb -> did not [X]
    
            """
        patterns = [
            [{"LOWER": "is"}],
            [{"LOWER": "are"}],
            [{"LOWER": "may"}],
            [{"LOWER": "could"}],
            [{"LOWER": "should"}],
            [{"LOWER": "was"}],
            [{"LOWER": "were"}],
            [{"LOWER": "have"}],
            [{"LOWER": "can"}],
            [{"LOWER": "has", "POS": "AUX"}],
            [{"TAG": "VVZ"}],
            [{"TAG": "VVN"}],
            [{"TAG": "VBD"}],
            [{"TAG": "VBZ"}],
            [{"TAG": "VBP"}],
        ]

        def get_neg_sub(index, tok, pos):
            if index > 9:
                #print("yielding: ", replacements[index], "+ ", (tok if type(tok) is str else tok.lemma_) )
                if type(tok) is str:
                    lemma = getLemma(tok,pos)
                    if lemma:
                        lemma = lemma[0]
                    else:
                        lemma = tok
                else:
                    lemma = tok.lemma_
                return replacements[index] + " " + lemma
            else:
                return replacements[index]

        replacements = [
            "is not",
            "are not",
            "may not",
            "could not",
            "should not",
            "was not",
            "were not",
            "have not",
            "cannot",
            "has not",
            "does not",
            "did not",
            "did not",
            "does not",
            "do not",
        ]

        matcher = Matcher(nlp.vocab)
        for i, p in enumerate(patterns):
            matcher.add(key=i,patterns=[p],greedy='FIRST')


        #df2 = dataset.iloc[dl,:]
        #df2.to_csv('testfeverneg.csv')

        print("stage -1")
        #print(dataset)
        claims = nlp.pipe(dataset['claim'],batch_size=16)
        evidences = nlp.pipe(dataset['evidence'],disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer","wsd"],batch_size=16)
        cla_list = []
        done_ids = []
        nsns, nsas, nsbs = [],[],[]

        print("stage0")

        ev_l = (any(p._.negex for p in z.ents) for z in evidences)
        del evidences
        nsb = None
        count = 0
        claim_last = None
        #ant_cache={}
        ant_cache = pickle.load(open("caches/ant_cache11600.p", "rb"))
        for claim, ev_neg, label_raw, id in zip(claims,ev_l, dataset['label'],dataset['id']):
            print("//////////////////////////////")
            print("CLAIM: ", claim, " LABEL: ", label_raw)
            if count > 0 and nsb:
                if claim.text == claim_last:
                    done_ids.append(id)
                    nsbs.append(nsb)
                    nsas.append(nsa)
                    nsns.append(nsn)
                    print("skipping duplicate: ", claim)
                    count += 1

                    continue

            if count % 100 == 0:
                done_dict = {'id': done_ids, 'new_claims_both': nsbs, 'new_claims_antonym': nsas,
                             'new_claims_negation': nsns}
                dfdone = pd.DataFrame(done_dict)
                dfdone.to_csv('caches/outputat'+str(count)+'.csv', index=False)
                pickle.dump(ant_cache, open("caches/ant_cache_train"+str(count)+".p", "wb"))

            count+=1
            print(count, "of ", len(dataset))

            gc.collect()
            claim_neg = any(q._.negex for q in claim.ents)
            label = LABEL_DICT[label_raw]
            #print(claim, ev_neg, label_raw, id, asizeof(claim), asizeof(ev_neg),asizeof(label_raw),asizeof(id))


            #print("newcl: ", claim)

            candidate = True
            if label == -1 and ((claim_neg and ev_neg) or (not claim_neg and not ev_neg)):
                candidate = False

            elif label == 1 and (claim_neg and not ev_neg) or (not claim_neg and ev_neg):
                candidate = False

            elif label == 0:
                candidate = False

            print("CLAIMNEGATED: ", claim_neg, "EV NEG: ", ev_neg, "CANDIDATE:", candidate)
            if candidate and not claim_neg and id not in cla_list:
                gc.collect()

                cla_list.append(id)
                #print(claim)
                verb = get_first_verb(claim)
                if not verb:
                    print("XXX - NO VERB - : ", list(tok.pos_ for tok in claim))
                    continue
                obj, prep_flag = get_obj(claim)



                new_sent_both, new_sent_neg, new_sent_ant = [], [], []
                verb_empty = False
                from_cache_flag = False
                for iii, aq_fake in enumerate([verb,obj,prep_flag,0]):
                    if iii == 3:
                        aq = (obj if (verb.lemma_ == "be" or verb_empty==True) else verb)
                    else:
                        aq = aq_fake
                    if not aq: continue
                    # Negate:
                    p = matcher(claim)
                    if p:
                        ind, start, end = p[0]

                        if iii == 3:
                            replacement = None

                        else:
                            replacement = antonym_exists(aq)
                            print(aq, "-> ", replacement)
                            if aq.text in ant_cache:
                                replacement = ant_cache[aq.text]
                                print("retrieved from cache")
                                from_cache_flag = True
                            if not replacement or (aq == verb and verb.lemma_ == "be"):
                                continue
                            if aq.text not in ant_cache:
                                inflected = getInflection(replacement,aq.tag_)
                                if inflected:
                                    replacement = inflected[0]

                        if not from_cache_flag:
                            print("(", iii, ")", claim, "  ENTER ANTONYM OF:", aq, " Current: ", replacement)
                            new_x = input("+ to accept, - to skip, or type a new one to reject")
                            if not new_x or new_x[0] == "-":
                                verb_empty = True
                                #print("setting VE")
                                continue
                            elif new_x[0] != "+":
                                replacement = new_x
                        if not replacement: continue
                    else:
                        #print("NO NEGATION: ", claim, replacement.tag_)
                        break

                    if replacement and aq.text not in ant_cache:
                        ant_cache[aq.text] = replacement if type(replacement) is str else replacement.text
                        #print("Set cache for",aq.text," to ", replacement)

                    for tok in claim:
                        if tok.i == end-1:
                            if tok.i == aq.i:
                                neg_sub=get_neg_sub(ind, replacement,tok.pos_)+tok.whitespace_

                            else:
                                neg_sub = get_neg_sub(ind, tok,tok.pos_)+tok.whitespace_

                            #print("//////////////")
                            new_sent_both.append(neg_sub)
                            new_sent_neg.append(get_neg_sub(ind, tok,tok.pos_)+tok.whitespace_)
                            #new_sent_ant.append(tok.text_with_ws)

                        elif type(aq) is Span and tok.i in (range(aq.start,aq.end-1)):
                            pass
                        else:
                            if tok.i == aq.i:
                                new_sent_both.append(replacement+tok.whitespace_)
                                new_sent_ant.append(replacement + tok.whitespace_)
                                new_sent_neg.append(tok.text_with_ws)

                            else:
                                new_sent_both.append(tok.text_with_ws)
                                new_sent_ant.append(tok.text_with_ws)
                                new_sent_neg.append(tok.text_with_ws)


                    break

                gc.collect()
                #print("NSB: ", new_sent_both)
                if new_sent_both:
                    claim_last = claim.text
                    done_ids.append(id)
                    nsb = "".join(new_sent_both)
                    nsa = "".join(new_sent_ant)
                    nsn = "".join(new_sent_neg)
                    print("NEGATION ONLY: ", nsn, "NEG+ANT: ", nsb)

                    nsbs.append(nsb)
                    nsas.append(nsa)
                    nsns.append(nsn)
                else:
                    print("SKIPPED FOR SOME REASON: ", claim.text)

        #Contains all the neg-claim candidates for conversion
        done_dict = {'id':done_ids, 'new_claims_both':nsbs,'new_claims_antonym':nsas,'new_claims_negation':nsns}
        dfdone = pd.DataFrame(done_dict)
        dfdone.to_csv('done_neg_overlap_fever_dev_auto_UG.csv',index=False)
        return


    def convert_back(concept):
        if '/c/en/' not in concept: return None
        k=concept.split('/c/en/')[1]
        return k[:-1].replace('_',' ')


    def antonym_exists(tok):
        return "ANT_PLACEHOLDER"
        if not tok: return

        if tok.dep_ == 'prep' and tok.lower_ in preplookup:
            #Find antonym from - https://www.clres.com/db/prepstats.
            #print("PREP CHANGE:", tok, " -> ",preplookup[tok.lower_],  "\n\n\n\n\n\n")
            return preplookup[tok.lower_]
        elif tok._.offset:
            #print(tok, tok._.offset)

            #Detect special cases where wordnet failed but conceptnet didn't - here no resolving is needed.
            if tok._.offset[0] == '*':
                pos_map = {'VERB':'v','PROPN':'n','NOUN':'n'}
                offset = tok._.offset
                offset_r = '_'+pos_map.get(tok.pos_,'a')
                concept = tok._.offset[1:]
            else:
                offset_r = tok._.offset.split(':')[1]
                offset = offset_r[:-1]+'-'+offset_r[-1]
                concept = wnconvert.get(offset,None)
            if concept: #If mapping of synset to conceptnet concept found:
                #check for antonym for word itself,
                #if none found, then find synonyms for the concept in sim_map,
                #and check those for antonyms.
                #print(offset," -> CONCEPT MAPPING: ", concept)
                antonym = antsearch.get(concept+'/'+offset_r[-1]+'/')
                if antonym:
                    #print("ANTONYM FOUND: ", antonym)
                    return convert_back(antonym)
                else:
                    #print("NO ANTONYM FOUND, TRYING SYNONYMS")
                    synonyms = simconv.get(concept+'/'+offset_r[-1],None)
                    if synonyms:
                        #print("SYNONYMS FOUND: ",synonyms)
                        for syn in synonyms:
                            if tok.lemma_ in syn:
                                antonym = antsearch.get(syn+'/', None)
                                if antonym:
                                    return convert_back(antonym)

                        for syn in synonyms:
                            antonym=antsearch.get(syn+'/',None)
                            if not antonym:
                                antonym = antsearch.get(concept + '/a/')
                            if antonym: return convert_back(antonym)
                        #print("NO antonyms found for synonyms")

                    else:
                        antonym = ppdb_s.get(tok.lemma_,None)
                        if not antonym: antonym = ppdb_s.get(tok.text,None)

                        if not antonym and offset_r[-1] == 'v':
                            print("trying with /a/")
                            antonym = antsearch.get(concept + '/a/')
                        if not antonym:
                            print("trying stripped")
                            antonym = antsearch.get(concept + '/')
                        if antonym: return convert_back(antonym)


        #Fall back to wordnet if none in conceptnet
        w2 = (tok._.synset)
        if w2 is not None and w2.lemmas():
            for lemma in w2.lemmas():
                ants = lemma.antonyms()
                if ants:
                    return ants[0].name()
        return


    def get_obj(doc):
        for tok in doc:
            if 'obj' in tok.dep_:
                return (tok, prep_check(tok))
        #Where no object, likely an intransitive verb so look for subject:
        for tok in doc:
            if 'subj' in tok.dep_ and tok.ent_iob_=='O':
                return (tok, prep_check(tok))
        return (None, None)

    def prep_check(tok):
        verb = tok.head if (tok.pos_ == "AUX") else tok
        for c in verb.children:
            if c.dep_ == 'prep':
                return c
        return


    def get_first_verb(doc):
        for tok in doc:
            if tok.pos_ == 'VERB':
                return tok
        for tok in doc:
            if tok.pos_ == 'AUX':
                return tok


        return


def sbert(row):
    sentences1 = [row['claim']]
    sentences2 = [row['evidence']]
    embeddings1 = transf.encode(sentences1)
    embeddings2 = transf.encode(sentences2)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return float(cosine_scores[0][0])

def non_stop_jaccard(row):
    d1 = nlp_min(row['claim'])
    d2 = nlp_min(row['evidence'])
    s1 = set()
    s2 = set()
    for tok in d1:
        if not tok.is_stop:
            s1.add(tok.text)
    for tok in d2:
        if not tok.is_stop:
            s2.add(tok.text)
    return len(s1.intersection(s2))/min(len(s1),len(s2))

def tfidf_sim(row):
    samps = [row['claim'],row['evidence']]
    tfm = sktfid.fit_transform(samps)
    cosi = skcs(tfm,tfm)
    return(cosi[0][1])

def generate_overlap(dataset):
    """Does high word overlap result in assumed SUP?
    Also test if high word overlap + a negation word (in exclusive-either hyp or text) is assumed as REF

    Adversaries + what label they *should* be (i.e. the one the model shouldn't get if weak to the adversary)
    Hyp + text high overlap - REF/NEI
    Hyp + text (very low) overlap - SUP

    """
    dataset['tfidf'] = dataset.apply(tfidf_sim,axis=1)
    dataset['sbert'] = dataset.apply(sbert,axis=1)

    print("STDEV", dataset['sbert'].std(), dataset['sbert'].mean())

    dfx = pd.DataFrame({'label':dataset['label'],'sbert':dataset['sbert']})
    dfx.to_csv('sbert_vals2.csv')

    dataset['nsjacc'] = dataset.apply(non_stop_jaccard,axis=1)

    #High overlaps:
    tfid_filt_high = dataset[(dataset['tfidf']>1.5) & dataset['label'].isin(['REFUTES','NOT ENOUGH INFO'])]
    sbert_filt_high = dataset[(dataset['sbert']>1.7) & dataset['label'].isin(['REFUTES','NOT ENOUGH INFO'])]
    ns_jacc_filt_high = dataset[(dataset['nsjacc']>1.5) & dataset['label'].isin(['REFUTES','NOT ENOUGH INFO'])]

    # High overlaps:
    tfid_filt_low = dataset[(dataset['tfidf'] < 0.1) & dataset['label'].isin(['REFUTES','SUPPORTS'])]
    sbert_filt_low = dataset[(dataset['sbert'] < 0.4) & dataset['label'].isin(['NOT ENOUGH INFO'])]
    ns_jacc_filt_low = dataset[(dataset['nsjacc'] < 0.1) & dataset['label'].isin(['REFUTES','SUPPORTS'])]

    tfids = pd.concat([tfid_filt_low,tfid_filt_high])
    sbertids = pd.concat([sbert_filt_low,sbert_filt_high])
    nsjaccids = pd.concat([ns_jacc_filt_low,ns_jacc_filt_high])

    #tfids.to_csv('fever-dev-tf-id_overlap_examples.csv')
    #sbertids.to_csv('fever-dev-sbert_overlap_examples.csv')
    #nsjaccids.to_csv('fever-dev-nonstop-jaccard_overlap_examples.csv')



    return

def get_obj_head(tok):
    z=tok
    while z.pos_ not in ["PROPN","NOUN"]:
        z=z.head
        if z.head == z:
            return None
    return z


def get_encapsulating_nc(tok):
    for nc in tok.doc.noun_chunks:
        if tok.i in range(nc.start,nc.end):
            for z in nc:
                if z.pos_ in ["NOUN","PROPN"]:
                    return nc
    else:
        return None

def needs_conv(num,rela,case,label):
    """

    :param num: Number present in claim
    :param rela: Number in evidence
    :param case: Which of cases 1-4
    :param label: Overall label example.
    :return: -1 if ranges don't work with any cases below,
    1 if they do but need to be modified to make the label fit,
    0 if they already fit.
    """

    # Adversaries: Ranges not overlapping
    # Case 1: ev:not less + cl:less -> SUP (return 0)
    # Case 2: ev:not more + cl:more -> SUP (return 0)
    # +Convert any case 1/2 which -> REF/NEI to go -> SUP (return 1)
    #Case 1/2 but overlapping ranges -> Drop (Are too difficult to modify were 1 to be returned)

    # Adversaries: Ranges overlapping
    # Case 3: ev:more + cl:more -> NEI/REF (return 0)
    # Case 4: ev:less + cl:less -> NEI/REF (return 0)
    # +Convert any case 3/4 which -> SUP to go -> NEI/REF (return 1)
    #Case 3/4 but non-overlapping ranges (return 2 - need to modify the range whilst preserving label)

    try:
        i1 = float(next(tok.text_with_ws for tok in num if tok.like_num)) #claim
        i2 = float(rela.text) #evidence
    except:
        return -1

    if any(x in num.text.lower() for x in ['under', 'fewer', 'less']):
        i1 = i1 * 0.9999999
    if any(x in num.text.lower() for x in ['more', 'over']):
        i2 = i2 * 0.9999999

    #print("claim(i1)", i1, "ev(i2)", i2)
    if case < 3:
        if (case==1 and i1 <= i2) or (case==2 and i2<=i1):
            #Non-overlapping ranges
            if label== "SUPPORTS":
                return 1
            else:
                return 0
        else:
            return -1

    else:
        if (case==3 and i1<=i2) or (case==4 and i2<=i1):
            #Overlapping ranges
            if label == "SUPPORTS":
                return 0
            else:
                return 1
        else:
            return 2

def generate_num_mismatch():
    """Nearly all the vitaminC examples containing numbers go from
     e.g. 'More than 150' to 'more than 149' for the SUP or 'less than 149' for the REF.
    As with named_ent_overweight, are systems learning to match that pattern alone?
    i.e. are they seeing the 'x people have died' evidence and assuming SUP when seeing a 'less than x+1 people have died' claim?

    Adversarial examples:
    where (more than) means either 'more than', or just nothing (e.g. "more than 10" or just "10"),
    x is a value or within 10% of one in whichever direction would make it correct
    e.g. '...31 men have been charged...' with '...more than 30 men have been arrested...' = REF/NEI (REF for reasons
    contained in the rest of the text or NEI if there isn't).


    format: evidence, claim, label:
    (more than) x , less/fewer than x, e.g. more than 10,000 deaths / less than 9500 recoveries - model should  NOT assume REF just because the ranges don't overlap here.

    (less than) x, more than x,
    (more than) x, more than x,
    (less than) x, less/fewer than x,

    Then perform the same with x as dates.

    Can test here for range mismatches - e.g. does evidence: 22 men died + claim: more than 25 were killed correctly
    yield REF but claim: more than 20 were killed correctly yield SUP - VitC Already seems to test for this though?

    Instead can check if range match/non-match is taken as automatic indicator of sup/ref

    """


    def matwrap(row):
        doc = Doc(vocab=nlp.vocab, words=[tok for tok in row['evidence'].split(' ') if tok != ''])
        return matcher(doc)

    def matwrap2(row):
        doctoks = [tok for tok in row['claim'].split(' ') if tok != '']
        doc = Doc(vocab=nlp.vocab, words=doctoks)

        doctoks2 = [tok for tok in row['evidence'].split(' ') if tok != '']
        doc2 = Doc(vocab=nlp.vocab, words=doctoks2)

        claim_match = matcher2(doc)
        evidence_match = matcher2(doc2)

        if len(evidence_match) > 1:
            if evidence_match[0][0] != evidence_match[1][0]:
                return (-1,evidence_match)

        #Ranges not overlapping -> SUP
        #Case 1: ev:not less + cl:less
        #Case 2: ev:not more + cl:more

        # Ranges overlapping -> REF
        #Case 3: ev:more + cl:more
        #Case 4: ev:less + cl:less

        if not claim_match:
            return (0,None)

        if claim_match[0][0] == 3:
            #Case 1
            if (not evidence_match or evidence_match[0][0] != 3):
                return (10,claim_match)

            #Case 4
            elif row['label'] in ['REFUTES','NOT ENOUGH INFO'] and (not evidence_match or evidence_match[0][0] == 3):
                return (40,claim_match)

        elif claim_match[0][0] == 2:

            #Case 2
            if (not evidence_match or evidence_match[0][0] != 2):
                return (20,claim_match)

            #Case 3
            elif row['label'] in ['REFUTES','NOT ENOUGH INFO'] and (not evidence_match or evidence_match[0][0] == 2):
                return (30,claim_match)

        return (0,claim_match)

    df = pd.read_json('VitaminC/data/vitaminc/test.jsonl',lines=True)
    patterns = [[{"LIKE_NUM": True}]]
    nlp = spacy.load('en_core_web_lg')
    matcher = Matcher(nlp.vocab)

    patterns_less = [[{"ORTH": {"IN":['under','fewer','less']}},{"OP": "?"},{"LIKE_NUM": True}]]
    patterns_more = [[{"ORTH": {"IN":['more','over']}},{"OP": "?"},{"LIKE_NUM": True}]]

    matcher2 = Matcher(nlp.vocab)

    matcher.add(1,patterns=patterns, greedy='FIRST')
    matcher2.add(2,patterns=patterns_more, greedy='FIRST')
    matcher2.add(3,patterns=patterns_less, greedy='FIRST')


    df['numerical_claim'] = df.apply(matwrap,axis=1)
    df['more_less'] = df.apply(matwrap2,axis=1) #2 = More, 3=Less
    df2=df[(df['numerical_claim'].map(len) >0) & df['more_less'].apply(lambda x: x[0] > 0) & (df['claim'].apply(lambda x: 'review' not in x))]

    #df2.to_csv('numerical_vitc_dev.csv')

    samp=df2[df2['label'] != 'NOT ENOUGH INFO']
    #samp = df2.head(2)
    nlp3 = spacy.load('en_core_web_trf')
    docs = nlp3.pipe(samp['claim'])
    docs2 = nlp3.pipe(samp['evidence'])
    index = 0
    new_claims = []
    new_labels = []
    ids = []
    for claim, evidence, row_raw in zip(docs,docs2,samp.iterrows()):
        index+=1
        row = row_raw[1]
        print("///////////",row['label'],"///////// (",str(index), "of ", str(len(samp)))
        print(claim, list(claim.noun_chunks),list(k.root for k in claim.noun_chunks), list(claim.ents))
        print(evidence, list(evidence.noun_chunks))
        print("case:", str(row['more_less'][0])[0])
        h1 = list(get_obj_head(evidence[i[1]:i[2]][0]) for i in row['numerical_claim']) #Checking evidence
        h2 = list((get_obj_head(claim[i[1]:i[2]][0])) for i in row['more_less'][1]) #Checking claim
        h3 = []
        for i, x in enumerate(h1):
            for j, y in enumerate(h2):
                if x is not None and y is not None:
                    if x.lemma == y.lemma:
                        h3.append((row['numerical_claim'][i],y,row['more_less'][1][j]))

        if not h3:
            h1 = list(get_encapsulating_nc(evidence[i[1]:i[2]][0]) for i in row['numerical_claim'])  # Checking evidence
            h2 = list((get_encapsulating_nc(claim[i[1]:i[2]][0])) for i in row['more_less'][1])  # Checking claim
            for i, x in enumerate(h1):
                for j, y in enumerate(h2):
                    if x is not None and y is not None:
                        if x.root.lemma == y.root.lemma:
                            h3.append((row['numerical_claim'][i], y, row['more_less'][1][j]))
        if not h3:
            print("panic h3")
            continue

        hh = h3[0]

        num_to_modify = claim[hh[2][1]:hh[2][2]]
        relative_to = evidence[hh[0][1]:hh[0][2]]
        noun = hh[1]
        print(num_to_modify, "/", relative_to)
        conv_needed = needs_conv(num_to_modify,relative_to,(row['more_less'][0])/10,row['label'])
        if conv_needed == -1:
            continue
        elif conv_needed == 1:
            print("Keep range but flip label")

            new_noun = input(noun.text+": ")
            if len(new_noun) and new_noun[0] == "¬":
                new_claim = new_noun[1:]
            elif new_noun == "+" or type(noun) is Span:
                continue
            else:
                new_claim = "".join(tok.text_with_ws if tok.i != noun.i else new_noun + tok.whitespace_ for tok in claim)
            label = input("new label (1=sup, 0=nei, -1=ref): ")
            print(label, ": ",new_claim)
            new_claims.append(new_claim)
            new_labels.append(label)
            ids.append("m1"+row['unique_id'])


        elif conv_needed ==2:
            print("PRESERVE LABEL ", row['label'], "  and Adapt range")
            new_int = input(num_to_modify[-1])
            new_noun = input(noun.text)
            if len(new_noun) and new_noun[0] == "¬":
                new_claim = new_noun[1:]
            elif new_noun == "+":
                continue
            else:
                n_toks = []
                for tok in claim:
                    if tok.i == noun.i:
                        n_toks.append(new_noun+ " ")
                    elif tok.i == num_to_modify.end - 1:
                        n_toks.append(new_int+ " ")
                    else:
                        n_toks.append(tok.text_with_ws)
                new_claim = "".join(n_toks)
            new_claims.append(new_claim)
            label = input("new label (0=nei, -1=ref): ")
            new_labels.append(label)
            ids.append("m2"+row['unique_id'])



        else:
            new_claims.append(claim)
            new_labels.append(row['label'])
            ids.append(row['unique_id'])

        if index % 100 == 0:
            print("saving")
            new_df = pd.DataFrame({'id': ids, 'new_claim': new_claims, 'new_label': new_labels})
            #new_df.to_csv('num_mismatch.csv')

        #Adversaries: Ranges not overlapping -> SUP
        #Case 1: ev:not less + cl:less
        #Case 2: ev:not more + cl:more
        #+Convert any case 1/2 which -> REF/NEI to go -> SUP

        #Adversaries: Ranges overlapping -> NEI/REF
        #Case 3: ev:more + cl:more
        #Case 4: ev:less + cl:less
        # +Convert any case 3/4 which -> SUP to go -> NEI/REF


    new_df = pd.DataFrame({'id':ids,'new_claim':new_claims,'new_label':new_labels})
    #new_df.to_csv('num_mismatch.csv')
    return


def filter_ev(ev):
    ret_ev = []
    labels = []
    for e in ev:
        if e.label_ == "PERSON" or e.label_ not in labels:
            if 'tomato' in e.text.lower():
                ret_ev.append("RT ratings aggregator")
            else:
                labels.append(e.label_)
                ret_ev.append(e)
    return ret_ev

def filter_ev2(doc):
    ev = doc.ents
    ret_ev = []
    labels = []
    reject_list = []
    for e in ev:
        print(e, e.label_)
        if e.label_ == "PERSON" or e.label_ not in labels:
            if 'tomato' in e.text.lower():
                ret_ev.append("RT ratings aggregator")
            else:
                labels.append(e.label_)
                if e.label_ == "WORK_OF_ART":
                    ret_ev.append("'"+e.text+"'")
                else:
                    ret_ev.append(e.text)
        elif 'tomato' not in e.text.lower() and e.label_ != "CARDINAL":
            reject_list.append(e)

    if len(ret_ev) < 5:
        reject_list.extend(list(x.root for x in doc.noun_chunks if 'tomato' not in x.text.lower()))
        for x in reject_list:
            if x.text not in ret_ev: ret_ev.append(x.text)
            if len(ret_ev) == 5: return ret_ev

    return ret_ev

def filter_ev3(doc):
    print("ENTS: ",doc.ents)
    print("ENTy: ", list(e.label_ for e in doc.ents))
    if 'tomato' in doc.text.lower():
        return filter_ev2(doc)
    else:
        return filter_ev(doc.ents)


def generate_named_ent_overweight():
    """
    The effective converse to word overlap - are there examples being marked as SUP just because they contain overlapping named ents
    e.g.  "Vladimir Putin has recently opened a florists' with 'Putin put forward plans for an attack on Ukraine'.
    Both mention Putin but have clearly very different semantics.

    Adversaries:
    Hyp+text sharing a named ent but have an NEI relation.
    """
    from keytotext import pipeline
    nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
    nlp2 = pipeline('k2t')
    df = pd.read_json('VitaminC/data/vitaminc/test.jsonl',lines=True)
    #df = pd.read_csv('ungram.csv')

    print(df)
    #df = df[df['label'] != "NOT ENOUGH INFO"]
    df.reset_index()
    df['new_claims'] = None
    coldex = df.columns.get_loc('new_claims')
    coldec = df.columns.get_loc('claim')


    nlp3 = spacy.load('en_core_web_trf')
    docs2 = nlp3.pipe(["Steve sought $ 210 million in damages and was ongoing as of 2016 "])
    lend2 = len(df)

    for i, evidence in enumerate(docs2):
        print("///////////////")
        print(df.iloc[i,coldec])
        print(evidence)

        #print(evidence.ents)
        #print(list(evidence.noun_chunks))
        #print(filter_ev(evidence.ents))

        if i%100 == 0:
            print(str(i), " of ", lend2)
            if i%5000 == 0:
                df.to_csv('caches/overweighting'+str(i)+'.csv')

        #nc2 = list(nc.root for nc in evidence.noun_chunks)
        #nc3 = set()
        #for x in nc2: nc3.add(x)
        #for x in filter_ev(evidence.ents): nc3.add(x)
        nc4 = filter_ev3(evidence)
        newval = nlp(nc4).replace("RT ", "Rotten Tomatoes ").split(". ")[0]
        print("NEWV ", newval)
        if "VERB" not in (list(t.pos_ for t in nlp3(newval))) or "olympic athlete" in newval:
            df.iloc[i,coldex] = nlp2(nc4).replace("RT ", "Rotten Tomatoes ").split(". ")[0]
        else:
            df.iloc[i, coldex] = newval
    #df.to_csv('ungramnew.csv')


    return


#todo later
def augmented_insensitivity():
    """
    Do slightly perturbed (but semantically unchanged) examples cause an (erroneous) change in judgement?
    https://textattack.readthedocs.io/en/latest/3recipes/augmenter_recipes.html

    This is related to the word overlap bias in that perturbation will reduce word overlap

    Adversaries:
    Perturbed-but-semantically-identical-hypothesis, text -> Original Label

    Alternatively, convert to AMR and back to text.
    """
    return


if __name__ == '__main__':
    dataset = get_dataset_stream('fever','dev')
    #ds2 = pd.read_csv('neg_overlap_fever_dev_sample.csv')
    #generate_neg_mismatch(dataset)
    #generate_num_mismatch()
    #generate_named_ent_overweight()
    generate_overlap(dataset)