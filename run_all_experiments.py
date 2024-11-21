import os

repeat_times = 2
for dataset in ['pascal', 'coco']:
    for fold in range(4):        
        cmd = f'python -u main.py --dataset {dataset} --fold {fold} --weights checkpoints/{dataset}_fold{fold}.ckpt'
        os.system(cmd)
