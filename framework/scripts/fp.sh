#!bin/bash

config_dir="config/"
csv_dir="out/"
config_file='baseline.yaml'
missing_rate_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
missing_type_list=('whole' 'part')
dataset_name_list=('Wisconsin')
# dataset_name_list=('Cora' 'Wisconsin' 'chameleon' 'flickr')

filling_method='feature_propagation'
gpu_idx=8

csv_file='feature_propagation.csv'
lr=0.75

for missing_type in "${missing_type_list[@]}"; do
    echo "dataset,$missing_type" >> $PWD/$csv_dir$csv_file;
    for dataset_name in "${dataset_name_list[@]}"; do
        all_outputs=()
        for missing_rate in "${missing_rate_list[@]}"; do
            result=()
            mapfile -t result < <(python run.py \
                --config-file $PWD/$config_dir$config_file\
                --opts dataset_name "$dataset_name" missing_type "$missing_type"\
                missing_rate "$missing_rate" filling_method "$filling_method"\
                gpu_idx "$gpu_idx" lr "$lr")
            # echo "$ntype","$noise","$gamma","$lr","$momentum","${result[-4]}","${result[-3]}","${result[-2]}","${result[-1]}" >> $PWD/$csv_dir$csv_file

            all_outputs+=("${result[-1]}")
        done
    
        # Join all outputs into a single string, separated by a delimiter (e.g., comma)
        output_row=$(IFS=,; echo "${all_outputs[*]}")
        dataset_output="$dataset_name,$output_row"
        # Write the concatenated output to the CSV file
        echo "$dataset_output" >> "$PWD/${csv_dir}${csv_file}"
    done
done
