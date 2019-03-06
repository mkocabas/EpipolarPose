import logging
import time

import numpy as np
import torch

from lib.utils.img_utils import trans_coords_from_patch_to_org_3d
from lib.core.integral_loss import get_result_func
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


def train_integral(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_data, batch_label, batch_label_weight, meta = data

        optimizer.zero_grad()

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_label_weight = batch_label_weight.cuda()

        batch_size = batch_data.size(0)
        # compute output
        preds = model(batch_data)

        loss = criterion(preds, batch_label, batch_label_weight)
        del batch_data, batch_label, batch_label_weight, preds

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), batch_size)
        del loss
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=batch_size/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)


def validate_integral(val_loader, model):
    print("Validation stage")
    result_func = get_result_func()

    # switch to evaluate mode
    model.eval()

    preds_in_patch_with_score = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            batch_data, batch_label, batch_label_weight, meta = data

            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

            # compute output
            preds = model(batch_data)
            del batch_data, batch_label, batch_label_weight


            preds_in_patch_with_score.append(result_func(256, 256, preds))
            del preds

        _p = np.asarray(preds_in_patch_with_score)

        # Dirty solution for partial batches
        if len(_p.shape) < 2:
            tp = np.zeros(((_p.shape[0] - 1) * _p[0].shape[0] + _p[-1].shape[0], _p[0].shape[1], _p[0].shape[2]))

            start = 0
            end = _p[0].shape[0]

            for t in _p:
                tp[start:end] = t
                start = end
                end += t.shape[0]

            _p = tp
        else:
            _p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))

        preds_in_patch_with_score = _p[0: len(val_loader.dataset)]

        return preds_in_patch_with_score


def eval_integral(epoch, preds_in_patch_with_score, val_loader, final_output_path, debug=False):
    print("Evaluation stage")
    # From patch to original image coordinate system
    imdb_list = val_loader.dataset.db
    imdb = val_loader.dataset

    preds_in_img_with_score = []

    for n_sample in range(len(val_loader.dataset)):
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], 256, 256,
                                              2000, 2000))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    name_value, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path, debug=debug)
    for name, value in name_value:
        logger.info('Epoch[%d] Validation-%s %f', epoch, name, value)

    return perf
