import os
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths

from refiner.data import Human36M
from refiner.model import get_model, weight_init
from refiner.utils import lr_decay, AverageMeter, save_ckpt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
    parser.add_argument('--load', type=str, default=None, help='path to load a pretrained checkpoint')
    parser.add_argument('--mode', type=str, default='train', help='mode: [train, test]')
    parser.add_argument('--num_epochs', type=int, default=200, help='num epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='# steps of lr decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96)
    args = parser.parse_args()

    return args

def train(model, train_dl, optimizer, glob_step, lr_now, criterion, args, logger):

    losses = AverageMeter()

    model.train()

    start = time.time()

    for i, (inp, tar) in enumerate(train_dl):
        glob_step += 1

        if glob_step % args.lr_decay == 0 or glob_step == 1:
            lr_now = lr_decay(optimizer, glob_step, args.lr, args.lr_decay, args.lr_gamma)

        inputs = inp.cuda()
        targets = tar.cuda()

        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs[0], targets) + criterion(outputs[1], targets)

        losses.update(loss.item(), targets.size(0))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()

    logger.info('Avg Loss: %.5f' % losses.avg)

    return glob_step, lr_now

def test(model, test_dl):

    model.eval()

    preds = []
    for i, (inp, tar) in enumerate(test_dl):
        inputs = inp.cuda()

        outputs = model(inputs)[-1]

        out_arr = outputs.cpu().detach().numpy()

        for j in range(out_arr.shape[0]):
            preds.append(out_arr[j])

    preds = np.asarray(preds)

    error = test_dl.dataset.evaluate(preds)

    return error

if __name__ == '__main__':
    args = parse_args()

    err_best = 1000

    log_dir = os.path.join('refiner/experiments', args.exp)
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(log_dir,'%s_log_%s.log'%(args.mode, time_str)),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model = get_model(weights=None)
    model = model.cuda()
    model.apply(weight_init)

    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load:
        logger.info(">>> loading ckpt from '{}'".format(args.load))
        ckpt = torch.load(args.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    train_dl = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=True),
        batch_size=64,
        shuffle=True,
        num_workers=8
    )

    test_dl = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=False),
        batch_size=64,
        shuffle=False,
        num_workers=8
    )

    logger.info("- done.")

    cudnn.benchmark = True

    if args.mode == 'train':

        logger.info("Starting training for {} epoch(s)".format(args.num_epochs))

        glob_step = 0
        lr_now = args.lr

        for epoch in range(args.num_epochs):
            logger.info('%s | %s | lr: %.6f' % (epoch, args.num_epochs, lr_now))

            glob_step, lr_now = train(model, train_dl, optimizer, glob_step, lr_now, criterion, args, logger)

            logger.info('Evaluation')

            error = test(model, test_dl)

            is_best = error < err_best
            err_best = min(error, err_best)

            save_ckpt({'epoch': epoch + 1,
                       'lr': lr_now,
                       'step': glob_step,
                       'err': error,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict()},
                      ckpt_path=log_dir,
                      is_best=is_best)
            if is_best:
                logger.info('Found new best, error: %s' % error)

    elif args.mode == 'test':
        test(model, test_dl)
    else:
        print('mode input error!')