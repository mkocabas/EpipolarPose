import argparse
import os
import pprint
import shutil
import _init_paths

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.function import train_integral
from lib.core.function import validate_integral, eval_integral
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger

import lib.core.integral_loss as loss
import lib.dataset as dataset
import lib.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int,
                        default=8)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    best_perf = 0.0

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose3d_resnet.get_pose_net(config, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)

    shutil.copy2(
        args.cfg,
        final_output_dir
    )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    loss_fn = eval('loss.'+config.LOSS.FN)
    criterion = loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM).cuda()

    # define training, validation and evaluation routines
    train = train_integral
    validate = validate_integral
    evaluate = eval_integral

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Resume from a trained model
    if not(config.MODEL.RESUME is ''):
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))

    # Choose the dataset, either Human3.6M or mpii
    ds = eval('dataset.'+config.DATASET.DATASET)

    # Data loading code
    train_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TRAIN_SET,
        is_train=True
    )
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        preds_in_patch_with_score = validate(valid_loader, model)
        acc = evaluate(epoch, preds_in_patch_with_score, valid_loader, final_output_dir, debug=config.DEBUG.DEBUG)

        perf_indicator = 500. - acc if config.DATASET.DATASET == 'h36m' or 'mpii_3dhp' or 'jta' else acc

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

if __name__ == '__main__':
    main()
