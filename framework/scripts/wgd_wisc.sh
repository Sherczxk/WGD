#!bin/bash

config_dir="config/"
csv_dir="out/"
config_file='wgd_wisc.yaml'
missing_rate_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# missing_rate_list=(0.1 0.5 0.9)
# missing_type_list=('whole' 'part')
# dataset_name_list=('Wisconsin' 'Cora' 'chameleon')

filling_method='wgd'
gpu_idx=8

layer_L_list=(2 4 6 8)
h_hop_list=(1 2 4 6 8)
# layer_L=2
# h_hop=2

lr_list=(0.5 0.1 0.01 0.001)
num_iters_list=(10)
num_projs_list=(100)
alpha_list=(0.5 1.0 2.0)
# lr=0.005
# alpha=0.5

dataset_name='Wisconsin'
missing_type='whole'

n_runs=10

csv_file='wgd_wisc.csv'


echo "dataset,h_hop,layer_L,lr,num_iters,num_projs,alpha,$missing_type" >> $PWD/$csv_dir$csv_file;
for lr in "${lr_list[@]}"; do
    for layer_L in "${layer_L_list[@]}"; do
        for h_hop in "${h_hop_list[@]}"; do
            for alpha in "${alpha_list[@]}"; do
                for num_iters in "${num_iters_list[@]}"; do
                    for num_projs in "${num_projs_list[@]}"; do
                        all_outputs=()
                        for missing_rate in "${missing_rate_list[@]}"; do
                            result=()
                            mapfile -t result < <(python run.py \
                                --config-file $PWD/$config_dir$config_file\
                                --opts dataset_name "$dataset_name" missing_type "$missing_type"\
                                missing_rate "$missing_rate" filling_method "$filling_method"\
                                gpu_idx "$gpu_idx" h_hop "$h_hop" layer_L "$layer_L"\
                                n_runs "$n_runs" lr "$lr" bary_comp_para.num_iters "$num_iters"\
                                bary_comp_para.num_projs "$num_projs" bary_comp_para.alpha "$alpha") 
                            # echo "$ntype","$noise","$gamma","$lr","$momentum","${result[-4]}","${result[-3]}","${result[-2]}","${result[-1]}" >> $PWD/$csv_dir$csv_file
                            echo "${result[-1]}"
                            all_outputs+=("${result[-1]}")
                        done
                    
                        # Join all outputs into a single string, separated by a delimiter (e.g., comma)
                        output_row=$(IFS=,; echo "${all_outputs[*]}")
                        dataset_output="$dataset_name,$h_hop, $layer_L, $lr, $num_iters, $num_projs,$alpha,$output_row"
                        # Write the concatenated output to the CSV file
                        echo "$dataset_output" >> "$PWD/${csv_dir}${csv_file}"
                    done
                done
            done
        done
    done
done
