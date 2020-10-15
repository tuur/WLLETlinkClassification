# Demo 

# output directory & task
out_dir='demo-output'
task=CR

# Data
train="data/artificial/Train/" # supervised training RC data
test="data/artificial/Test/" # RC test data
proxy="data/artificial/Raw/" # raw text unsupervised SG(LR) data

# To define GPUs (if available, e.g. 0)
export CUDA_VISIBLE_DEVICES="" 

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

 # Max number of training epochs (small for the demo)
 num_epochs=25

## RC (rnd init)
model_path=$out_dir/RC_rnd_init/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -init_with_pretrained_sg_w2v_embeddings 0
		
## RC (SG init)
model_path=$out_dir/RC_sg_init/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim

## RC (SG fixed)
model_path=$out_dir/RC_sg_fixed/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -fix_embeddings_at 0

## RC + SG
model_path=$out_dir/RC+SG/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -proxy SG -proxy_weight $pw

## RC + SGLR
model_path=$out_dir/RC+SGLR/
python run_experiment.py -task $task -bs $bs -num_epochs $num_epochs -model_dir $model_path -train_data $train -test_data $test -proxy_data $proxy -embedding_dim $emb_dim -lstm_dim $lstm_dim -proxy SGLR -proxy_weight $pwlr




