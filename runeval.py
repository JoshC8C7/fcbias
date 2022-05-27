
models = [
    "models/fxx_model",
    "tals/albert-base-vitaminc-fever",
    "tals/albert-base-vitaminc",
]

models=[]

for training_set in ['fever_only','vitc_fever','vitc_only']:
    for bias_method in ['poe','poe_anneal','reweight','reweight_anneal','conf_reg','conf_reg_anneal']:
        for bias_file in ['known_ent_overweight','shallow']:
            models.append('models/bias_trained/'+training_set+'/'+bias_method+'/'+bias_file)


adversaries_mine = [
"entity_overweighting_vitc_test_5000",
"fever-dev-nonstop-jaccard_overlap_new",
"fever-dev-sbert_overlap_new",
"fever-dev-tfidf_overlap_new",
"fever_dev_neg_overlap_semiauto_2000_antandneg",
"fever_dev_neg_overlap_semiauto_2000_negonly",
"neg_overlap_fever_dev_fullauto_ungrammatical",
"negation_overlap_fever_dev_manual",
"negation_overlap_fever_test_manual",
"numerical_mismatch_vitc_test_ungrammatical"
]


test_tasks = [
"vitaminc_real",
"vitaminc_synthetic",
"fever",
"fever_adversarial",
"fever_symmetric",
"fever_triggers",
"entity_overweighting",
"neg_overlap_manual",
"numerical_mismatch",
"sbert_overlap"
]

def get_cmd(mn):
    tasks = " ".join(test_tasks)
    cmd = "python scripts/fact_verification.py \
  --model_name_or_path "+mn+" \
  --test_tasks "+tasks+" \
  --data_dir data/final_eval_set \
  --max_seq_length 256 \
  --per_device_eval_batch_size 32 \
  --do_test \
  --output_dir results/auto_final/"+mn

    return cmd

def write_testing_cmd():
    cmds = list(get_cmd(model) for model in models)
    out_cmd = (" && ".join(cmds))
    with open("eval_gen.sh",'w') as f:
        f.write(out_cmd)

def read_test_results():
    import pandas as pd
    dict_list = []
    for p in models:
        model_res_dict = {}
        for task in test_tasks:
            pa = 'Vitaminc/results/bert_results/' + p+"/"+"test_results_"+task+".txt"
            pa = p+"/"+"test_results_"+task+".txt"

            res_df = pd.read_table(pa,sep='=',header=None)
            model_res_dict[task] = res_df.loc[1,1]
        dict_list.append(model_res_dict)
    d2 = pd.DataFrame(dict_list)
    d2.index = models
    print(d2)
    d2.to_csv('newresres.csv')

ld = {}

def get_losses():
    import json
    for p in models:
        pk = 'VitaminC/'+p+'/checkpoint-50000/trainer_state.json'
        try:
            with open(pk) as f:
                train_d = json.load(f)

                print(p)
                losses = list(k['loss'] for k in train_d['log_history'])
                ld[p] = losses

        except:
            print("skipping", p)

get_losses()

import pandas as pd
df = pd.DataFrame(ld)
df.to_csv('losses_alb.csv')