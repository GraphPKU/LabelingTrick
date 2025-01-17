# commands

$dataset can be selected from USAir, NS, PB, Yeast, Router, Power, Celegans, Ecoli.

python seal.py --dataset $dataset --model PGIN --runs 10 --num_hops $nh --node_label $nl 

python seal.py --dataset $dataset --model GIN --runs 10 --num_hops $nh --node_label $nl 

$nh = 1 if $dataset in ['PB', 'Ecoli'] else 2

For OGB datasets, the commands are as follows,

python seal.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --node_label $nl --model $model 
python seal.py --dataset ogbl-ddi  --hidden_channels 96  --lr 0.0001 --num_hops 1 --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_val --dynamic_test --train_percent 1 --node_label $nl --model $model
python seal.py --dataset ogbl-citation2 --num_hops 1 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1 --node_label $nl --model $model
python seal.py --dataset ogbl-ppa --num_hops 1 --use_feature --use_edge_weight --eval_steps 5 --epochs 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 5 --node_label $nl --model $model
