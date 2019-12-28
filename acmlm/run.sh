export export CUDA_VISIBLE_DEVICES=0

python fa_mlm.py \
	--corpus yelp \
	--do_train \
	--do_eval \
	--do_lower_case \
	--bert_model bert-base-uncased \
	--max_seq_length 30 \
	--train_batch_size 96 \
	--eval_batch_size 64 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir ./tmp/model
