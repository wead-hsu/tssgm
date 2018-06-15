save_dir='data/agnews/ag8000'

CUDA_VISIBLE_DEVICES=2 python3 clf.py \
	--model_path sssp.models.clf.basic_clf \
	--train_path ${save_dir}/labeled.data.idx \
	--valid_path ${save_dir}/valid.data.idx \
	--test_path ${save_dir}/test.data.idx \
	--save_dir 'results/ag/clf-lstm-8k-noemb' \
	--classifier_type "LSTM" \
	--num_classes 4 \
	--validate_every 500 \
	--vocab_size  23829 \
	#--embd_path ${save_dir}/embd.pkl \
	#--vocab_path ${save_dir}/vocab.pkl \
	#--fix_sent_len 300 \
	#--num_filters 300 \
	#--labels_path ${save_dir}/class_map.pkl \
	#--w_regfrobenius 0.001 \
	#--w_regdiff 0.00003\
	#--w_regl1 0.00006\
	#--filter_size 3 \
	#--batch_size 64 \
	#--fixirrelevant \
	#--init_from results/tmp4/rnn_test-16000
