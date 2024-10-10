# CUDA_VISIBLE_DEVICES=3 nohup python main_md17.py --test  --dataset benzene > Tune/benzene.test 2>&1 &
import subprocess
import time
import os


def work(dataset: str, gpu_id: str, label: str, task: str):
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python magnet-link.py --task {task}  --dataset {dataset} --label {label} --test  > {dataset.replace('/', '_')}.{label}.{task}.catpred.newtest 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def opt(dataset: str, gpu_id: str, label: str, task: str):
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python magnet-link.py --task {task} --dataset {dataset} --label {label} > {dataset.replace('/', '_')}.{task}.{label}.opt 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def wait(gpulist):
    while True:
        time.sleep(30)
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        load = [int(i) for i in sel]
        for i in range(len(load) // 2):
            if i not in gpulist:
                continue
            if sum(load[2 * i:2 * i + 2]) < 20000:
                return i


for i, ds in enumerate(["webkb/wisconsin", "cora_ml/"]):  # "webkb/cornell", "webkb/texas", "telegram/", "citeseer/" 
    for label in ["zo"]: 
        for task in ["existence"]:
            work(ds, i, label, task)