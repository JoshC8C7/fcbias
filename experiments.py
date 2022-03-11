import spacy
from spacy.matcher import Matcher
import json
import gc
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

NEG_GO = False
if NEG_GO:
    from ewiser.spacy.disambiguate import Disambiguator
    # noinspection PyUnresolvedReferences
    from negspacy.negation import Negex

    with open('final_similarities_cn.json') as f:
        simconv = json.load(f)
    preplookup = pd.read_csv('preps.csv', header=None, index_col=0, squeeze=True).to_dict()
    wnconvert = pd.read_csv('final-wn2cn.csv', header=None, index_col=0, squeeze=True).to_dict()
    antsearch = pd.read_csv('cn_antonym_map.csv', header=None, index_col=0, squeeze=True).to_dict()

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
    def generate_neg_mismatch(dataset):
        LABEL_DICT = {"REFUTES": -1, "SUPPORTS": 1, "NOT ENOUGH INFO": 0}

        wsd = Disambiguator('ewiser/ewiser.semcor+wngt.pt')
        wsd.to('cuda')
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("negex")
        wsd.enable(nlp, 'wsd')
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

        def get_neg_sub(index, tok):
            if index > 9:
                #print("yielding: ", replacements[index], "+ ", (tok if type(tok) is str else tok.lemma_) )
                return replacements[index] + " " + (tok if type(tok) is str else tok.lemma_)
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

        #dl = [60, 67, 83, 159, 220, 316, 320, 377, 386, 481, 484, 485, 591, 714, 778, 873, 889, 1139, 1158, 1278, 1308, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1348, 1351, 1352, 1353, 1405, 1408, 1442, 1471, 1496, 1509, 1556, 1675, 1731, 1775, 1787, 1790, 1828, 1861, 2010, 2011, 2051, 2055, 2107, 2145, 2154, 2188, 2189, 2309, 2415, 2425, 2454, 2467, 2527, 2528, 2574, 2583, 2694, 2747, 2748, 2751, 2814, 2832, 2846, 2965, 2966, 3028, 3203, 3204, 3331, 3345, 3354, 3370, 3376, 3396, 3398, 3420, 3461, 3492, 3494, 3550, 3591, 3669, 3720, 4036, 4073, 4125, 4294, 4349, 4449, 4491, 4552, 4598, 4692, 4777, 4807, 4848, 4862, 4897, 4924, 4949, 5059, 5190, 5203, 5300, 5411, 5442, 5443, 5457, 5495, 5577, 5755, 5758, 5778, 5849, 5907, 5921, 5922, 5959, 5983, 6095, 6203, 6242, 6261, 6335, 6396, 6535, 6536, 6537, 6575, 6584, 6585, 6596, 6632, 6699, 6705, 6829, 6896, 6897, 6918, 6969, 7118, 7175, 7202, 7287, 7297, 7404, 7405, 7415, 7416, 7516, 7558, 7559, 7703, 7710, 7711, 7767, 7768, 7782, 7901, 7903, 8007, 8029, 8107, 8153, 8154, 8155, 8156, 8200, 8261, 8273, 8292, 8318]

        #df2 = dataset.iloc[dl,:]
        #df2.to_csv('testfeverneg.csv')

        print("stage -1")
        #print(dataset)
        claims = nlp.pipe(dataset['claim'],batch_size=16)
        evidences = nlp.pipe(dataset['evidence'],disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer","wsd"],batch_size=16)
        cla_list = []
        done_ids = []
        done_sents = []

        print("stage0")

        ev_l = (any(p._.negex for p in z.ents) for z in evidences)
        del evidences
        for claim, ev_neg, label_raw, id in zip(claims,ev_l, dataset['label'],dataset.index):
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

            if candidate and not claim_neg and id not in cla_list:
                gc.collect()

                cla_list.append(id)
                #print(claim)
                verb = get_first_verb(claim)
                if not verb:
                    #print("XXX - NO VERB - : ", list(tok.pos_ for tok in claim))
                    continue
                obj, prep_flag = get_obj(claim)
                if not obj:
                    #print("XXX - NO OBJ - : ", list(tok.dep_ for tok in claim))
                    continue
                #print("V: ", verb, verb._.synset, "Obj: ", obj, obj._.offset)

                #if WSD fails, try a conceptnet search.
                if not obj._.offset:
                    pass
                    #todo Search for one in conceptnet, then instantiate as * + that

                new_sent = None
                for aq in [verb,obj,prep_flag]:
                    replacement = antonym_exists(aq)
                    if replacement:
                        # Negate:
                        p = matcher(claim)
                        if p:
                            ind, start, end = p[0]

                        else:
                            #print("NO NEGATION: ", claim, replacement.tag_)
                            break

                        new_sent = []
                        for tok in claim:
                            if tok.i == end-1:
                                if tok.i == aq.i:
                                    new_sent.append(get_neg_sub(ind, replacement)+tok.whitespace_)
                                else:
                                    new_sent.append(get_neg_sub(ind, tok)+tok.whitespace_)
                            elif tok.i in range(start,end-1):
                                pass
                            else:
                                if tok.i == aq.i:
                                    new_sent.append(replacement+tok.whitespace_)
                                else:
                                    new_sent.append(tok.text_with_ws)

                        break
                gc.collect()
                if new_sent:
                    done_ids.append(id)
                    aa = "".join(new_sent)
                    #print("DONE:", claim, "-> ", aa)
                    done_sents.append(aa)

        #Contains all the neg-claim candidates for conversion
        done_dict = {'id':done_ids, 'new_claims':done_sents}
        dfdone = pd.DataFrame(done_dict)
        dfdone.to_csv('output_negations.csv')
        return


    def convert_back(concept):
        if '/c/en/' not in concept: return None
        k=concept.split('/c/en/')[1]
        return k[:-1].replace('_',' ')


    def antonym_exists(tok):
        if not tok: return

        if tok.dep_ == 'prep' and tok.lower_ in preplookup:
            #Find antonym from - https://www.clres.com/db/prepstats.
            #print("PREP CHANGE:", tok, " -> ",preplookup[tok.lower_],  "\n\n\n\n\n\n")
            return preplookup[tok.lower_]
        elif tok._.offset:

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
                antonym = antsearch.get(concept,None)

                if not antonym:
                    #print("Not found under base, looking under: ",concept+'/'+offset_r[-1]+'/' )
                    antonym = antsearch.get(concept+'/'+offset_r[-1]+'/')
                if antonym:
                    #print("ANTONYM FOUND: ", antonym)
                    return convert_back(antonym)
                else:
                    #print("NO ANTONYM FOUND, TRYING SYNONYMS")
                    synonyms = simconv.get(concept,None)
                    if synonyms:
                        #print("SYNONYMS FOUND: ",synonyms)
                        for syn in synonyms:
                            antonym=antsearch.get(syn,None)
                            if antonym: return convert_back(antonym)
                        #print("NO antonyms found for synonyms")

                    else:
                        pass
                        #print("NO SYNONYMS FOUND")


        #Fall back to wordnet if none in conceptnet
        w2 = (tok._.synset)
        if w2 is not None and w2.lemmas():
            for lemma in w2.lemmas():
                ants = lemma.antonyms()
                if ants:
                    #print(tok, " -> ", ants[0])
                    return ants[0].name()
        return


    def get_obj(doc):
        for tok in doc:
            if 'obj' in tok.dep_:
                return (tok, prep_check(tok))
        #Where no object, likely an intransitive verb so look for subject:
        for tok in doc:
            if 'subj' in tok.dep_:
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





def generate_overlap():
    """Does high word overlap result in assumed SUP?
    Also test if high word overlap + a negation word (in exclusive-either hyp or text) is assumed as REF

    Adversaries + what label they *should* be (i.e. the one the model shouldn't get if weak to the adversary)
    Hyp + text high overlap - REF/NEI
    Hyp + text (very low) overlap - SUP
    Hyp + neg-text high overlap - SUP
    neg-hyp + text high overlap - SUP


    """



    return

def generate_num_mismatch():
    """Nearly all the vitaminC examples containing numbers go from
     e.g. 'More than 150' to 'more than 149' for the SUP or 'less than 149' for the REF.
    As with named_ent_overweight, are systems learning to match that pattern alone?
    i.e. are they seeing the 'x people have died' claim and assuming SUP when seeing 'less than x+1 people have died'?

    Adversarial examples:
    where (more than) means either 'more than', or just nothing (e.g. "more than 10" or just "10"),
    x is a value or within 10% of one in whichever direction would make it correct
    e.g. '...31 men have been charged...' with '...more than 30 men have been arrested...' = REF/NEI (REF for reasons
    contained in the rest of the text or NEI if there isn't).
    (more than) x , less/fewer than x, SUP/NEI e.g. more than 10,000 deaths / less than 9500 recoveries - model should  NOT assume REF just because the ranges don't overlap here.

    (less than) x, more than x, SUP/NEI
    (more than) x, more than x, REF/NEI
    (less than) x, less/fewer than x, SUP/NEI

    Then perform the same with x as dates.
    """

    return

def generate_named_ent_overweight():
    """
    The effective converse to word overlap - are there examples being marked as SUP just because they contain overlapping named ents
    e.g.  "Vladimir Putin has recently opened a florists' with 'Putin put forward plans for an attack on Ukraine'.
    Both mention Putin but have clearly very different semantics.

    Adversaries:
    Hyp+text sharing a named ent but have an NEI relation.
    """


    return


def augmented_insensitivity():
    """
    Do slightly perturbed (but semantically unchanged) examples cause an (erroneous) change in judgement?
    https://textattack.readthedocs.io/en/latest/3recipes/augmenter_recipes.html

    This is related to the word overlap bias in that perturbation will reduce word overlap

    Adversaries:
    Perturbed-but-semantically-identical-hypothesis, text -> Original Label
    """

    return


if __name__ == '__main__':
    dataset = get_dataset_stream('fever','dev')
