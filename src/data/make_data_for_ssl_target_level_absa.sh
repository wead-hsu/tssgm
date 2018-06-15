#python preprocess_for_target_level_absa.py
cd ..
cd ..
cd bilstmcrf

#mkdir ../data/se2014task06/bilstmcrf-rest
#mkdir data/rest
#mkdir results/rest
#python build_data.py --train_filename '../data/se2014task06/bilstmcrf-rest/train.seqtag.bo'\
#	--dev_filename '../data/se2014task06/bilstmcrf-rest/dev.seqtag.bo'\
#	--test_filename '../data/se2014task06/bilstmcrf-rest/test.seqtag.bo'\
#	--data_dir 'data/rest/'\
#	--save_dir 'results/rest/'
#
#python train.py --train_filename '../data/se2014task06/bilstmcrf-rest/train.seqtag.bo'\
#	--dev_filename '../data/se2014task06/bilstmcrf-rest/dev.seqtag.bo'\
#	--test_filename '../data/se2014task06/bilstmcrf-rest/test.seqtag.bo'\
#	--data_dir 'data/rest/'\
#	--save_dir 'results/rest/'

#python get_output_for_file.py --eval_filename '../data/extra-rest/eval.seqtag'\
#	--save_filename '../data/extra-rest/eval.seqtag.labeled'\
#	--data_dir 'data/rest/'\
#	--save_dir 'results/rest/'
#
#mkdir ../data/se2014task06/bilstmcrf-lapt
#mkdir data/lapt
#mkdir ../data/se2014task06/bilstmcrf-lapt
#python build_data.py --train_filename '../data/se2014task06/bilstmcrf-lapt/train.seqtag.bo'\
#	--dev_filename '../data/se2014task06/bilstmcrf-lapt/dev.seqtag.bo'\
#	--test_filename '../data/se2014task06/bilstmcrf-lapt/test.seqtag.bo'
#	--data_dir 'data/lapt/'\
#	--save_dir 'results/lapt/'
#
#python train.py --train_filename '../data/se2014task06/bilstmcrf-lapt/train.seqtag.bo'\
#	--dev_filename '../data/se2014task06/bilstmcrf-lapt/dev.seqtag.bo'\
#	--test_filename '../data/se2014task06/bilstmcrf-lapt/test.seqtag.bo'
#	--data_dir 'data/lapt/'\
#	--save_dir 'results/lapt/'
#
#python get_output_for_file.py --eval_filename '../data/extra-lapt/eval.seqtag'\
#	--save_filename '../data/extra-lapt/eval.seqtag.labeled'
#	--data_dir 'data/lapt/'\
#	--save_dir 'results/lapt/'
#
cd ..
cd src/data
mkdir ../../data/se2014task06/tabsa-rest
mkdir ../../data/se2014task06/tabsa-lapt
python postprocess_for_target_level_absa.py
