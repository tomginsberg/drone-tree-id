import argparse
import json
import os
from glob import glob

import cv2
import wandb

parser = argparse.ArgumentParser(description='Log some data')
parser.add_argument('--type', help='images or metrics')
parser.add_argument('--dest', help='location of output folder')

args = parser.parse_args()

if __name__ == '__main__':
    wandb.init(project='forest')

    if args.type == 'images':
        path = os.path.join(args.dest, 'vis', '**', '*')
        images = glob(path)
        for image in images:
            wandb.log({'/'.join(image.split('/')[-2:]): wandb.Image(cv2.imread(image)[:, :, ::-1])})

    elif args.type == 'metrics':
        metrics = os.path.join(args.dest, 'metrics.json')

        with open(metrics, 'r') as f:
            raw_data = f.read()

        for line in raw_data.split('\n')[:-1]:
            obj = json.loads(line)
            itr = obj['iteration']
            for k, v in obj.items():
                if k != 'iteration':
                    wandb.log({k: v}, step=itr)
    else:
        print('Please put metrics or images')
