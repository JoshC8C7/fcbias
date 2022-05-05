
models = [
    "tals/albert-base-vitaminc-fever",
    "tals/albert-base-vitaminc",
    "VitaminC/models/fxx_model"
]

models = [    "VitaminC/models/fxx_model"]

adversaries = [
    "emnlp-rules",
    "emnlp-sampled",
    "emnlp-searsfever",
    "emnlp-searssentiment",
    "emnlp-wnfever"
]


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


def get_cmd(mn,ds,msn):
    cmd = "python VitaminC/scripts/fact_verification.py \
  --model_name_or_path "+mn+" \
  --tasks_names " +ds+" \
  --data_dir VitaminC/data \
  --do_test \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_steps 50000 \
  --save_step 10000 \
  --overwrite_cache \
  --do_predict\
  --output_dir results/auto/"+msn+"/"+ds +" \
  "
    return cmd


cmds = []
for model in models:
    for dataset in adversaries_mine:
        model_save_name = model.split("/")[1]
        #print("launching",model,dataset)
        cmds.append(get_cmd(model,dataset,model_save_name))
        #print("done",model,dataset)





print(" && ".join(cmds))
"""
for model in models:
    print("////////"+model)
    for dataset in adversaries:
        model_save_name = model.split("/")[1]
        with open('results/auto/'+model_save_name+'/'+dataset+'/test_results_'+dataset+'.txt') as resf:
            print(dataset," : ",resf.readlines()[1].split("= ")[1])"""