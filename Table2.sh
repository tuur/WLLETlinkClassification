# Experimental settings as used for the THYME test set (Table 2*)
# *notice that weight initilizations are random so results might differ slightly for each run

# output directory & task
out_dir='table-2-output'
xml_regex=".*clin.*Temp.*"
task=CR

# To reproduce the results from Table 2:
train="data/real/THYME/Train+Dev/" # should contain the default THYME train and dev documents (one folder per id, containing txt + xml)
test="data/real/THYME/Test/" # should contain the default THYME test documents (one folder per id, containing txt + xml)
proxy="data/real/Raw/" # should contain the THYME train + dev documents and the MIMIC III subsection documents (raw .txt files)


# To define GPUs (if available)
export CUDA_VISIBLE_DEVICES=0 

#Hyperparameters

 # number of LSTM units
 lstm_dim=100 
 
 # word embedding dimension
 emb_dim=25 

 # SG weight
 pw=0.1 

 # SGLR weight
 pwlr=0.1

 # batch size
 bs=1024 

 # Max number of training epochs
 num_epochs=150

## RC (rnd init)
model_path=$out_dir/RC_rnd_init/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -init_with_pretrained_sg_w2v_embeddings 0 -file_regex_task $xml_regex
		
## RC (SG init)
model_path=$out_dir/RC_sg_init/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -file_regex_task $xml_regex

## RC (SG fixed)
model_path=$out_dir/RC_sg_fixed/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -fix_embeddings_at 0 -file_regex_task $xml_regex

## RC + SG
model_path=$out_dir/RC+SG/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -proxy SG -proxy_weight $pw -file_regex_task $xml_regex

## RC + SGLR
model_path=$out_dir/RC+SGLR/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -proxy SGLR -proxy_weight $pwlr -file_regex_task $xml_regex




