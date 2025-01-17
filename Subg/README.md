
To reproduce our results on synthetic datasets:
```
CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_one --use_seed --repeat 10 --device $gpu_id --dataset $dataset --model $model --test
```
where $dataset should be replace with the dataset you want to test, like `density`, `component`, `coreness`, and `cut_ratio`. $gpu_id should replace with the gpu you want to use. Set $gpu_id to -1 if you use cpu. $model can be selected from no (vanilla GNN), zo (zero-one labeling trick), pzo (subset zero-one labeling trick), opzo (one head subset zero-one labeling trick)

CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_one --use_seed --repeat 10 --device 0 --dataset density --model zo --test > density.zo.out &
CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_one --use_seed --repeat 10 --device 0 --dataset component --model zo --test > component.zo.out &
CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_one --use_seed --repeat 10 --device 0 --dataset coreness --model zo --test > coreness.zo.out &
CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_one --use_seed --repeat 10 --device 0 --dataset cut_ratio --model zo --test > cut_ratio.zo.out &


To reproduce our results on real-world datasets:

We have provided our SSL embeddings in ./Emb/. You can also reproduce them by
```
python GNNEmb.py --use_nodeid --device $gpu_id --dataset $dataset --name $dataset
```
Then 
```
python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device $gpu_id --dataset $dataset --model $model --test
```
where $dataset can be selected from em_user, ppi_bp, hpo_metab, and hpo_neuro.

CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device 0 --dataset em_user --model zo --test > em_user.zo.out &
CUDA_VISIBLE_DEVICES=6 python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device 0 --dataset ppi_bp --model zo --test > ppi_bp.zo.out &
CUDA_VISIBLE_DEVICES=5 python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device 0 --dataset hpo_metab --model zo --test > hpo_metab.zo.out &
CUDA_VISIBLE_DEVICES=6 python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device 0 --dataset hpo_neuro --model zo --test > hpo_neuro.zo.out &
