import cv2
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

from lib.utils.utils import calc_total_skeleton_length
from lib.utils.transforms import get_affine_transform
from lib.utils.augmentation import occlude_with_objects
from lib.core.integral_loss import get_joint_location_result, generate_joint_location_label
from lib.utils.triangulation import iterative_LS_triangulation
from lib.utils.prep_h36m import from_worldjt_to_imagejt
import random
import numpy as np
from easydict import EasyDict as edict

def get_default_augment_config():
    config = edict()
    config.scale_factor = 0.25
    config.rot_factor = 30
    config.color_factor = 0.2
    config.do_flip_aug = False # True

    config.rot_aug_rate = 0.6  #possibility to rot aug
    config.flip_aug_rate = 0.5 #possibility to flip aug
    return config


def do_augmentation():
    aug_config = get_default_augment_config()

    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    do_flip = aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + aug_config.color_factor
    c_low = 1.0 - aug_config.color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale


def fliplr_joints(_joints, _joints_vis, width, matched_parts):
    """
    flip coords
    joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    joints_vis: same as joints
    width: image width
    matched_parts: list of pairs
    """
    joints = _joints.copy()
    joints_vis = _joints_vis.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    # c = center.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg, occlusion_aug=True):
    # from h,w,c(OpenCV) to c,h,w
    tensor = cvimg.copy()
    tensor = np.transpose(tensor, (2, 0, 1))
    # from BGR(OpenCV) to RGB
    # tensor = tensor[::-1, :, :]
    # from int to float
    tensor = tensor.astype(np.float32)
    return tensor


def trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                                   scale=1.0, rot=0):
    coords_in_org = coords_in_patch.copy()
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=True)
    for p in range(coords_in_patch.shape[0]):
        coords_in_org[p, 0:2] = trans_point2d(coords_in_patch[p, 0:2], trans)
    return coords_in_org


def trans_coords_from_patch_to_org_3d(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height
                                      , rect_3d_width, rect_3d_height, scale=1.0, rot=0):
    res_img = trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                                             scale, rot)
    res_img[:, 2] = coords_in_patch[:, 2] / patch_width * rect_3d_width
    return res_img


def rescale_pose_from_patch_to_camera(preds_in_patch, target_bone_len, parent_ids):
    preds_in_patch_base_pelvis = preds_in_patch - preds_in_patch[0]
    skeleton_length = calc_total_skeleton_length(preds_in_patch_base_pelvis, parent_ids)
    rescale_factor = 1.0 * target_bone_len / skeleton_length
    preds_in_patch_base_pelvis = rescale_factor * preds_in_patch_base_pelvis
    return preds_in_patch_base_pelvis


def self_supervision(preds, meta):
    batch_size = preds.shape[0]

    preds_in_patch_with_score = get_joint_location_result(256, 256, preds)

    preds_in_img_with_score = []
    for n_sample in range(batch_size):
        kp = trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample],
                                               meta['center_x'][n_sample],
                                               meta['center_y'][n_sample],
                                               meta['width'][n_sample],
                                               meta['height'][n_sample],
                                               256, 256,
                                               2000, 2000,
                                               scale=meta['scale'][n_sample],
                                               rot=meta['rot'][n_sample]
                                               )
        preds_in_img_with_score.append(kp)

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)
    coords_3d_in_global_frame = triangulate(preds_in_img_with_score, meta)

    batch_label, batch_label_weight = get_batch_labels_from_global_coords(coords_3d_in_global_frame, meta)

    return batch_label, batch_label_weight


def triangulate(kps, meta):
    half_batch_size = kps.shape[0] // 2

    coords_3d_in_global_frame = []

    for idx in range(half_batch_size):
        u1, u2 = kps[idx, :, 0:2], kps[half_batch_size + idx, :, 0:2]

        P1, P2 = meta['projection_matrix'][idx].numpy(), \
                 meta['projection_matrix'][half_batch_size + idx].numpy()

        pt_3d, pt_vis = iterative_LS_triangulation(u1, P1, u2, P2)
        coords_3d_in_global_frame.append(pt_3d)

    coords_3d_in_global_frame = np.asarray(coords_3d_in_global_frame)
    coords_3d_in_global_frame = np.vstack([coords_3d_in_global_frame, coords_3d_in_global_frame])
    return coords_3d_in_global_frame


def get_batch_labels_from_global_coords(coords_3d_in_global_frame, meta):
    batch_size = coords_3d_in_global_frame.shape[0]

    batch_label = []
    batch_label_weight = []

    for i in range(batch_size):
        scale = meta['scale'][i]
        aug_rot = meta['rot'][i]
        c_x = meta['center_x'][i]
        c_y = meta['center_y'][i]
        bb_width = meta['width'][i]
        bb_height = meta['height'][i]

        keypoints, transf, rot, fl, c_p = coords_3d_in_global_frame[i], meta['T'][i].numpy(), meta['R'][i].numpy(), \
                                         meta['f'][i].numpy(), meta['c'][i].numpy()

        rect2d_l, rect2d_r, rect2d_t, rect2d_b, joints, pt_3d, joints_vis, pelvis3d = \
            from_worldjt_to_imagejt(keypoints.shape[0], rot, keypoints, transf, fl, c_p, 2000., 2000.)

        trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, 256, 256, scale, aug_rot, inv=False)

        for n_jt in range(len(joints)):
            joints[n_jt, 0:2] = trans_point2d(joints[n_jt, 0:2], trans)
            joints[n_jt, 2] = joints[n_jt, 2] / (2000. * scale) * 256.

        label, label_weight = generate_joint_location_label(256., 256., joints, joints_vis)

        batch_label.append(label)
        batch_label_weight.append(label_weight)

    return np.asarray(batch_label, dtype=np.float32), np.asarray(batch_label_weight, dtype=np.float32)


def get_single_patch_sample(img_path, center_x, center_y, width, height,
                            joints, joints_vis, flip_pairs, parent_ids,
                            patch_width, patch_height, rect_3d_width, rect_3d_height, mean, std,
                            do_augment, label_func, depth_in_image=False, occluder=None, DEBUG=False):
    # 1. load image
    cvimg = cv2.imread(
        img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)

    img_height, img_width, img_channels = cvimg.shape

    # 2. get augmentation params
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.0, 0, False, [1.0, 1.0, 1.0]

    # 3. generate image patch
    img_patch_cv, trans = generate_patch_image_cv(cvimg, center_x, center_y, width, height, patch_width, patch_height,
                                                  do_flip, scale, rot)
    image = img_patch_cv.copy()
    image = image[:, :, ::-1]

    if occluder:
        image = occlude_with_objects(image, occluder)

    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    # apply normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    # 4. generate patch joint ground truth
    # flip joints and apply Affine Transform on joints
    if do_flip:
        joints, joints_vis = fliplr_joints(joints, joints_vis, img_width, flip_pairs)

    for n_jt in range(len(joints)):
        joints[n_jt, 0:2] = trans_point2d(joints[n_jt, 0:2], trans)
        if depth_in_image:
            joints[n_jt, 2] = joints[n_jt, 2] / (width * scale) * patch_width
        else:
            joints[n_jt, 2] = joints[n_jt, 2] / (rect_3d_width * scale) * patch_width

    # 5. get label of some type according to certain need
    label, label_weight = label_func(patch_width, patch_height, joints, joints_vis)

    return img_patch, label, label_weight, scale, rot


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped