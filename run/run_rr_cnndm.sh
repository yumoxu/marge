export TASK_NAME=rr
export MODEL_NAME=rr_cnndm
export PROJ_ROOT=~/shiftsum

export DATASET=multinews
export MARGE_DATASET=marge_l2020-ratio-reveal_0.0
export DATA_DIR=${PROJ_ROOT}/data/${DATASET}/${MARGE_DATASET}

export OUTPUT_DIR=${PROJ_ROOT}/model/${MODEL_NAME}
export LOG_DIR=${PROJ_ROOT}/log/${MODEL_NAME}

export python=${PROJ_ROOT}/bin/python
export python_file=${PROJ_ROOT}/src/rr/run.py

$python ${python_file} \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --max_seq_length 320 \
    --per_gpu_train_batch_size=16  \
    --per_gpu_eval_batch_size=128  \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_steps 10000 \
    --metric rouge_2_f1 \
    --rouge_c 0.15 \
