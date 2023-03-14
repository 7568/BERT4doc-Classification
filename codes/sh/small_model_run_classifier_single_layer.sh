data_type='tabular_to_textual'
#data_type='tabular_to_word'
bert_model='uncased_L-12_H-768_A-12'
#bert_model='cased_L-24_H-1024_A-16'
MODELS_1=(
         ["1"]=0,1
         ["2"]=2,3
         ["3"]=4,5
         ["4"]=6,7
         ["5"]=4,5
          )
printf "\n\n----------------------------------------------------------------------------\n"

cd ../fine-tuning
for item in  5;
  do
    rm -rf ./result/${bert_model}/${data_type}/cross_validation_$[${item}-1]/*
    printf "\n----------------------------------------------------------------------------\n"
    printf 'CUDA_VISIBLE_DEVICES %s ' "${MODELS_1[$item]}" 'data_type :' ${data_type}
    printf "\n----------------------------------------\n"
    CUDA_VISIBLE_DEVICES=${MODELS_1[$item]}
    python run_classifier_single_layer.py --log_to_file_name ${data_type}_${bert_model}_cross_validation_$[${item}-1] --task_name imdb --do_train --do_eval --do_lower_case --data_dir /home/liyu/data/tabular-data/adult/${data_type}/cross_validation_$[${item}-1] --vocab_file /home/liyu/data/${bert_model}/vocab.txt --bert_config_file /home/liyu/data/${bert_model}/bert_config.json --init_checkpoint /home/liyu/data/${bert_model}/pytorch_model.bin --max_seq_length 512 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./result/${bert_model}/${data_type}/cross_validation_$[${item}-1]/ --seed 42 --layers 11 10 --trunc_medium -1 & echo $! >> ../sh/pid/${bert_model}/${data_type}/$[${item}-1].pid
  done
cd ../sh
