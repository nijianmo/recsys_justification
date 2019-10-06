export TASK_NAME=justification

python bert_run_classifier.py \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--do_lower_case \
	--data_dir ./$TASK_NAME \
	--bert_model bert-base-uncased \
	--max_seq_length 64 \
	--train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir ./tmp/$TASK_NAME/
