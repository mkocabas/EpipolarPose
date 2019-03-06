import math

import numpy as np
import torch
from torch.nn import functional as F

from lib.utils.transforms import transform_preds
from lib.utils.cameras import load_cameras
from lib.utils.cameras import Camera


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def get_per_joint_error(config, output_hm, output_depth, meta, cams, glob, centered=True):
    """
    :param config: configuration object
    :param output_hm: predicted 2d heatmap, shape=(N x num_joints x output_shape x output_shape)
    :param output_depth: predicted depth map, shape=(N x num_joints x depth_res)
    :param meta: meta values
    :param cams: cam dictionary
    :return: per_joint_error: mean error in mm
    """

    batch_size = output_hm.shape[0]

    c = meta['center'].numpy()
    s = meta['scale'].numpy()
    gt_joints = meta['joints_3d'].numpy()
    subjects = meta['subject'].numpy()
    cam_ids = meta['cam_id'].numpy()

    images_dir = meta['image']

    n_joints = config.MODEL.NUM_JOINTS

    # Get predicted 2D joint coordinates
    preds, _ = get_final_preds(config, output_hm, c, s)

    # Get predicted depth coordinates in mm
    preds_depth = get_depth_preds(config, output_depth)

    gt_depth = meta['depth'].numpy()
    # gt_depth = get_depth_preds(config, gt_depth)


    error = 0.
    for i in range(batch_size):
        gt_joints_3d = gt_joints[i]
        file_name = images_dir[i]
        cam_params = cams[(subjects[i], cam_ids[i])]
        kps = preds[i]

        cam = Camera(cam_params)

        jj, dZ, _, _, _ = cam.project_point_radial(gt_joints_3d)

        # Unproject points from image plane to camera frame
        pred_joints_3d = cam.unproject_pts(kps, dZ[0] + preds_depth[i])
        pred_joints_3d = cam.camera_to_world_frame(pred_joints_3d)

        if centered:
            # Make joints hip oriented
            pred_joints_3d = pred_joints_3d - pred_joints_3d[0]
            gt_joints_3d = gt_joints_3d - gt_joints_3d[0]

        gt_joints_3d = gt_joints_3d.reshape((-1))
        pred_joints_3d = pred_joints_3d.reshape((-1))

        sqerr = (gt_joints_3d - pred_joints_3d) ** 2
        dists = np.zeros((n_joints))

        dist_idx = 0
        for k in np.arange(0, n_joints * 3, 3):
            # Sum across X,Y, and Z dimenstions to obtain L2 distance
            dists[dist_idx] = np.sqrt(np.sum(sqerr[k:k + 3]))
            dist_idx += 1
        e = np.mean(dists)
        error += e

        gt_joints_3d = gt_joints_3d.reshape((17, -1))
        pred_joints_3d = pred_joints_3d.reshape((17, -1))

    per_joint_error = error / batch_size
    return per_joint_error


def get_depth_preds(cfg, pred):
    """
    :param cfg: configuration object
    :param pred_depth: predicted depth map, shape=(N x num_joints x depth_res)
    :return: depth: distance from camera center in mm, shape=(N x num_joints)
    """
    batch_size = pred.shape[0]

    num_joints = cfg.MODEL.NUM_JOINTS
    depth_res = cfg.MODEL.DEPTH_RES
    depth_range = cfg.DATASET.DEPTH_RANGE
    depth = np.zeros((batch_size, num_joints))

    pred = pred.reshape((batch_size, num_joints, depth_res))

    for i in range(batch_size):
        for j in range(num_joints):
            d = softargmax1d(pred[i,j])
            d = d / float(depth_res) - 0.5
            d *= depth_range
            depth[i,j] = d

    return depth

def softargmax1d(inp):
    exp = np.exp(inp)
    i = np.arange(exp.shape[0])
    return np.sum(i * (exp / exp.sum()))