import argparse
import pprint
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
from lib.core.function import validate_integral, eval_integral
from lib.core.integral_loss import L1JointLocationLoss
from lib.utils.utils import create_logger

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
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose3d_resnet.get_pose_net(config, is_train=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    validate = validate_integral
    evaluate = eval_integral

    # Start from a model
    if not(config.MODEL.RESUME is ''):
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))

    ds = eval('dataset.' + config.DATASET.DATASET)

    # Data loading code
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    preds_in_patch_with_score = validate(valid_loader, model)
    acc = evaluate(0, preds_in_patch_with_score, valid_loader, final_output_dir, config.DEBUG.DEBUG)

    print(acc)


if __name__ == '__main__':
    main()