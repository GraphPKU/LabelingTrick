
To reproduce our results on synthetic datasets:
```
python PlabelMain.py --use_one --use_seed --use_maxzeroone --repeat 10 --device $gpu_id --dataset $dataset --model $model
```
where $dataset should be replace with the dataset you want to test, like density, component, coreness, and cut_ratio. $gpu_id should replace with the gpu you want to use. Set $gpu_id to -1 if you use cpu. $model can be selected from no (vanilla GNN), zo (zero-one labeling trick), pzo (subset zero-one labeling trick), opzo (one head subset zero-one labeling trick)


To reproduce our results on real-world datasets:

We have provided our SSL embeddings in ./Emb/. You can also reproduce them by
```
python GNNEmb.py --use_nodeid --device $gpu_id --dataset $dataset --name $dataset
```
Then 
```
python PlabelMain.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device $gpu_id --dataset $dataset --model $model
```
where $dataset can be selected from em_user, ppi_bp, hpo_metab, and hpo_neuro.
