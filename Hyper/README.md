# prepare dataset
```
mkdir data
python preprocess.py
```

# reproduce results
```
python main.py --dataset $dataset --model $model --test
```
$model can be label for zero-one labeling trick model and plabel for zero-one subset labeling trick

$dataset can be selected from ["DAWN", "email-Eu", "NDC-classes", "NDC-substances", "threads-ask-ubuntu", "threads-math-sx", "tags-ask-ubuntu", "tags-math-sx"]

# ablation
```
python main.py --dataset $dataset --model label --test --no_label
```