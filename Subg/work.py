# CUDA_VISIBLE_DEVICES=3 nohup python main_md17.py --test  --dataset benzene > Tune/benzene.test 2>&1 &
import subprocess
import time
import os


def work(dataset: str, gpu_id: str, label: str):
    cmd = f"python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} --model {label} > {dataset}.{label}.opt 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def test(dataset: str, gpu_id: str, label: str):
    cmd = f"python PlabelMain.py --use_nodeid --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} --model {label} --test > {dataset}.{label}.test 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def testsyn(dataset: str, gpu_id: str, label: str):
    cmd = f"python PlabelMain.py --use_one --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} --model {label} --test > {dataset}.{label}.test2 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def wait():
    while True:
        time.sleep(20)
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        load = [int(i) for i in sel]
        for i in range(len(load) // 2):
            if sum(load[2 * i:2 * i + 2]) < 1000:
                return i


for ds in ["cut_ratio", "coreness", "density"]:#, 
    for label in  ["no",  "zo"]: # ["opzo", "pzo"]: 
        dev = wait()
        testsyn(ds, dev, label)
