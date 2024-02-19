import sys

import wandb
from time import sleep
import os

def init_wandb(project_name, model_name, config, **wandb_kwargs):
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    while True:
        try:
            wandb_run = wandb.init(
                project=project_name, name=model_name, save_code=True,
                config=config, **wandb_kwargs,
                )
            break
        except Exception as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            sleep(1)
            print('retrying..', file=sys.stderr)
    return wandb_run

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)