import os
import torch

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_ckpt(state, ckpt_path, is_best=True):
    if is_best:
        file_path = os.path.join(ckpt_path, 'best.pth.tar')
        torch.save(state, file_path)
    else:
        file_path = os.path.join(ckpt_path, 'last.pth.tar')
        torch.save(state, file_path)