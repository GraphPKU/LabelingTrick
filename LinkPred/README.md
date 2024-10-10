# small datasets

$dataset can be selected from USAir, NS, PB, Yeast, Router, Power, Celegans, Ecoli

python seal.py --dataset $dataset --model PGIN --runs 10 --num_hops $nh --node_label $nl 

python seal.py --dataset $dataset --model GIN --runs 10 --num_hops $nh --node_label $nl 

$nh = 1 if $dataset in ['PB', 'Ecoli'] else 2
