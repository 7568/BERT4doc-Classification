DATA_TYPE='tabular_to_textual'
#DATA_TYPE='tabular_to_word'
#bert_model='uncased_L-12_H-768_A-12'
bert_model='uncased_L-24_H-1024_A-16'
MODELS_1=(
         ["1"]=0,1
         ["2"]=2,3
         ["3"]=4,5
         ["4"]=6,7
         ["5"]=4,5
          )
GPUS_0=(
      ["tabular_to_textual"]=0,1,2,3
#      ["tabular_to_word"]=4,5,6,7
        )

printf "\n\n----------------------------------------------------------------------------\n"

cd ../fine-tuning
for item in  3;
  do
    rm -rf ./result/${bert_model}/${DATA_TYPE}/cross_validation_$[${item}-1]/*
    printf "\n----------------------------------------------------------------------------\n"
    printf 'CUDA_VISIBLE_DEVICES %s \t DATA_TYPE : %s \n' ${GPUS_0[$DATA_TYPE]}  ${DATA_TYPE}
    printf 'log path : ../fine-tuning/log/%s/%s/%s \n' ${DATA_TYPE} ${bert_model} cross_validation_$[${item}-1]
    printf "\n----------------------------------------\n"
#    CUDA_VISIBLE_DEVICES=${MODELS_1[$item]}
    CUDA_VISIBLE_DEVICES=${GPUS_0[$DATA_TYPE]} python run_classifier_single_layer.py --log_to_file_name ${DATA_TYPE}_${bert_model}_cross_validation_$[${item}-1] --task_name imdb --do_train --do_eval --do_lower_case --data_dir /home/liyu/data/tabular-data/adult/${DATA_TYPE}/cross_validation_$[${item}-1] --vocab_file /home/liyu/data/${bert_model}/vocab.txt --bert_config_file /home/liyu/data/${bert_model}/bert_config.json --init_checkpoint /home/liyu/data/${bert_model}/pytorch_model.bin --max_seq_length 512 --train_batch_size 24 --learning_rate 1e-5 --num_train_epochs 30.0 --output_dir ./result/${bert_model}/${DATA_TYPE}/cross_validation_$[${item}-1]/ --seed 42 --layers 15 14 --trunc_medium -1 & echo $! >> ../sh/pid/${bert_model}/${DATA_TYPE}/$[${item}-1].pid
  done
cd ../sh
