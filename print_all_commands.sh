oracle_array=( \
    'troglitazone_rediscovery' \
    'celecoxib_rediscovery' \
    'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
    'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \
    'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
    'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop' \
    'qed' \
    'drd2' 'jnk3' 'gsk3b'
)

for seed in 1 2 3 4 5 ; do
    for oracle in "${oracle_array[@]}" ; do
        res_name="tanimoto-bo__${oracle}__${seed}"
        echo python main.py \
            --method=tanimoto_gpbo --oracle=$oracle \
            --out_file ./bo_results/${res_name}.json \
            --seed=$seed \
            --budget=10000 \> ./logs/${res_name}.log

    done
done
