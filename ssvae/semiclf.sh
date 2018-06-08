save_dir='data/agnews/ag8000'

CUDA_VISIBLE_DEVICES=cpu0 python3 semiclf.py \
	--model_path sssp.models.semiclf.semiclf_aaai17 \
	--classifier_type 'LSTM' \
	--train_label_path ${save_dir}/labeled.data.idx \
	--train_unlabel_path ${save_dir}/unlabeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--vocab_path ${save_dir}/vocab.pkl \
	--embd_path ${save_dir}/embd.pkl \
	--save_dir 'results/ag/semiclf-ag8k-lstm-nosample-emb-dropoutfix-cellclipeverywhere-hfix-nodivideklw' \
	--num_classes 4 \
	--num_pretrain_steps 0000 \
	--vocab_size 23829 \
	--sample_unlabel 'False' \
	#--use_weights
