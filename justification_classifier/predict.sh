export TASK_NAME=justification

python bert_predict_classification.py \
	--task_name $TASK_NAME \
	--do_eval \
	--do_lower_case \
	--data_dir ./data \
        --config_file ./tmp/$TASK_NAME/bert_config.json \
        --model_file ./tmp/$TASK_NAME/pytorch_model.bin \
	--bert_model bert-base-uncased \
	--max_seq_length 64 \
	--eval_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ./tmp/$TASK_NAME/
