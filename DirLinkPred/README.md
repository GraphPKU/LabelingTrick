Environment:
```
pip install torch_geometric_signed_directed
```

python magnet-link.py --task existence  --dataset $dataset --label $label --test 

$dataset can be selected from "webkb/wisconsin", "cora_ml/", "webkb/cornell", "webkb/texas", "telegram/", "citeseer/" 
$label can be selected from no (no labeling trick), zo (set zero-one), and dizo (zero-one subset labeling trick with onehead routine)

