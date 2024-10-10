import subprocess
import time

def work(device, dataset, model):
    cmd = f"CUDA_VISIBLE_DEVICES={device} nohup python main.py --dataset {dataset} --model {model} --test > out/{dataset}.{model}.out 2>&1 &"
    print(cmd)
    subprocess.call(cmd, shell=True)

def nolabel(device, dataset):
    cmd = f"CUDA_VISIBLE_DEVICES={device} nohup python main.py --dataset {dataset} --model label --test --no_label > out/{dataset}.nolabel.out 2>&1 &"
    print(cmd)
    subprocess.call(cmd, shell=True)

def wait():
    while True:
        time.sleep(40)
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        load = [int(i) for i in sel]
        for i in range(len(load) // 2):
            if sum(load[2 * i:2 * i + 2]) < 15000:
                return i


ds = ["DAWN", "email-Eu", "NDC-classes", "NDC-substances", 
    "threads-ask-ubuntu", "threads-math-sx", "tags-ask-ubuntu", "tags-math-sx"]

for dst in ds:
    dev  = wait()
    nolabel(dev, dst)
    for model in ["label", "plabel"]:
        dev  = wait()
        work(dev, dst, model)