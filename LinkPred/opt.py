import optuna
import subprocess

def experiment(trial: optuna.Trial):
    dim = trial.suggest_int("dim", 32, 256, 32)
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3])
    num_layer = trial.suggest_int("num_layer", 1, 3)
    ret = subprocess.check_output(f"CUDA_VISIBLE_DEVICES=2 python seal.py --num_layers {num_layer} --lr {lr} --hidden_channels {dim} --dataset ogbl-ddi --num_hops 1 --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_val --dynamic_test --train_percent 1 --node_label dizo --model PGIN", shell=True)
    ret = str(ret, encoding="utf-8")
    ret = ret.split("$")
    ret = float(ret[1])
    return ret



stu = optuna.create_study("sqlite:///ddi.db", study_name="ddi", direction="maximize", load_if_exists=True)
stu.optimize(experiment, 200)