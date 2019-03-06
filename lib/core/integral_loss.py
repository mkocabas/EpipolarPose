import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math

def weighted_mse_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_smooth_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    diff = input - target
    abs = torch.abs(diff)
    out = torch.where(abs<1., 0.5*diff**2, abs-0.5)

    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(float(x_dim)), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(float(y_dim)), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(float(z_dim)), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z

def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    else:
        assert 0, 'Not Implemented!'
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class L2JointLocationLoss(nn.Module):
    def __init__(self, num_joints,size_average=True, reduce=True, norm=False):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        print(num_joints)

        pred_jts = softmax_integral_tensor(preds, self.num_joints, self.output_3d, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class L1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class SmoothL1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(SmoothL1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_smooth_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

def get_loss_func(config):
    if config.loss_type == 'L1':
        return L1JointLocationLoss(config.output_3d)
    elif config.loss_type == 'L2':
        return L2JointLocationLoss(config.output_3d)
    else:
        assert 0, 'Error. Unknown heatmap type {}'.format(config.heatmap_type)

def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis

def reverse_joint_location_label(patch_width, patch_height, joints):
    joints = joints.reshape((joints.shape[0] // 3, 3))

    joints[:, 0] = (joints[:, 0] + 0.5) * patch_width
    joints[:, 1] = (joints[:, 1] + 0.5) * patch_height
    joints[:, 2] = joints[:, 2] * patch_width
    return joints

def get_joint_location_result(patch_width, patch_height, preds):
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]

    hm_depth = hm_width
    num_joints = preds.shape[1] // hm_depth

    pred_jts = softmax_integral_tensor(preds, num_joints, True, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

def get_label_func():
    return generate_joint_location_label

def get_result_func():
    return get_joint_location_result

def merge_flip_func(a, b, flip_pair):
    # NOTE: flip test of integral is implemented in net_modules.py
    return a

def get_merge_func(loss_config):
    return merge_flip_func
