import json
import os
from sys import argv

import wandb

if __name__ == '__main__':
    metrics = os.path.join(argv[1], 'metrics.json')

    with open(metrics, 'r') as f:
        raw_data = f.read()

    wandb.init(project='forest')

    for line in raw_data.split('\n')[:-1]:
        obj = json.loads(line)
        itr = obj['iteration']
        for k, v in obj.items():
            if k != 'iteration':
                wandb.log({k, v}, step=itr)
