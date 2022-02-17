

#Available models:
"""
From VitC:
Albert-base-vitaminc - huggingface/tals/albert-base-vitaminc
Albert-base-vitaminc-fever huggingface/tals/albert-base-vitaminc-fever
Albert-base with fever only
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
fever_adversarial
fever_symmetric
fever_triggers


"""

def train_albert(prop_fever):
    #Trains a vitC albert model, stores it as alb-{prop_fever}
    #where prop_fever is proportion of fever.
    # E.g. alb-0_1 is 10% (0.1) fever, 90% (0.9) vitc.
    return


def predict_albert(data_dir, model):
    #Use {model} to predict data from data_dir, yields summary metrics.
    return

def generate_neg_mismatch():
    return


def generate_antonym():
    return

def generate_overlap():
    return

def generate_num_mismatch():
    return

def generate_named_ent_overweight():
    return




if __name__ == '__main__':
    print()