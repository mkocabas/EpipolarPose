import os
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from lib.utils.prep_h36m import compute_similarity_transform
import logging

MPII_NAMES = [''] * 15
MPII_NAMES[0] = 'RFoot'
MPII_NAMES[1] = 'RKnee'
MPII_NAMES[2] = 'RHip'
MPII_NAMES[3] = 'LHip'
MPII_NAMES[4] = 'LKnee'
MPII_NAMES[5] = 'LFoot'
# MPII_NAMES[6]  = 'Hip'
MPII_NAMES[6] = 'Thorax'
MPII_NAMES[7] = 'Neck/Nose'
MPII_NAMES[8] = 'Head'
MPII_NAMES[9] = 'RWrist'
MPII_NAMES[10] = 'RElbow'
MPII_NAMES[11] = 'RShoulder'
MPII_NAMES[12] = 'LShoulder'
MPII_NAMES[13] = 'LElbow'
MPII_NAMES[14] = 'LWrist'

logger = logging.getLogger(__name__)


class Human36M(Dataset):
    def __init__(self, is_train):
        fname = 'refiner/data/train.pkl' if is_train else 'refiner/data/valid.pkl'

        self.is_train = is_train

        self.data, self.labels, self.data_mean, \
        self.data_std, self.labels_mean, self.labels_std = self.get_db(fname)

        logger.info('loaded %s samples from %s' % (len(self.data), fname))

    def get_db(self, fname):
        with open(fname, 'rb') as anno_file:
            anno = pkl.load(anno_file)

        data = np.asarray(anno['inp'], dtype=np.float32).reshape(len(anno['inp']), -1)
        labels = np.asarray(anno['out'], dtype=np.float32).reshape(len(anno['out']), -1)

        # Remove hip joint
        data = np.delete(data, np.s_[18:21], axis=1)
        labels = np.delete(labels, np.s_[18:21], axis=1)

        if os.path.exists('refiner/data/norm.pkl'):
            with open('refiner/data/norm.pkl', 'rb') as f:
                data_mean, data_std, labels_mean, labels_std = pkl.load(f)
        elif self.is_train:
            data_mean, data_std = data.mean(axis=0), data.std(axis=0)
            labels_mean, labels_std = labels.mean(axis=0), labels.std(axis=0)

            with open('refiner/data/norm.pkl', 'wb') as f:
                pkl.dump((data_mean, data_std, labels_mean, labels_std), f)

        data = (data - data_mean) / data_std

        if self.is_train:
            labels = (labels - labels_mean) / labels_std

        if self.is_train:
            rnd = np.random.permutation(labels.shape[0])
            data, labels = data[rnd], labels[rnd]

        return data, labels, data_mean, data_std, labels_mean, labels_std

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def evaluate(self, preds):
        preds = preds * self.labels_std + self.labels_mean

        preds = preds.reshape((preds.shape[0], -1, 3))

        dist = []
        dist_align = []
        dist_14 = []
        dist_14_align = []
        dist_x = []
        dist_y = []
        dist_z = []
        dist_per_joint = []

        j14 = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]


        for i in range(len(preds)):

            pre_3d_kpt = preds[i]
            gt_3d_kpt = self.labels[i].reshape((-1, 3))

            joint_num = pre_3d_kpt.shape[0]

            # align
            _, Z, T, b, c = compute_similarity_transform(gt_3d_kpt, pre_3d_kpt, compute_optimal_scale=True)
            pre_3d_kpt_align = (b * pre_3d_kpt.dot(T)) + c

            diff = (gt_3d_kpt - pre_3d_kpt)
            diff_align = (gt_3d_kpt - pre_3d_kpt_align)

            e_jt = []
            e_jt_align = []
            e_jt_14 = []
            e_jt_14_align = []
            e_jt_x = []
            e_jt_y = []
            e_jt_z = []

            for n_jt in range(0, joint_num):
                e_jt.append(np.linalg.norm(diff[n_jt]))
                e_jt_align.append(np.linalg.norm(diff_align[n_jt]))
                e_jt_x.append(np.sqrt(diff[n_jt][0] ** 2))
                e_jt_y.append(np.sqrt(diff[n_jt][1] ** 2))
                e_jt_z.append(np.sqrt(diff[n_jt][2] ** 2))

            for jt in j14:
                e_jt_14.append(np.linalg.norm(diff[jt]))
                e_jt_14_align.append(np.linalg.norm(diff_align[jt]))

            dist.append(np.array(e_jt).mean())
            dist_align.append(np.array(e_jt_align).mean())
            dist_14.append(np.array(e_jt_14).mean())
            dist_14_align.append(np.array(e_jt_14_align).mean())
            dist_x.append(np.array(e_jt_x).mean())
            dist_y.append(np.array(e_jt_y).mean())
            dist_z.append(np.array(e_jt_z).mean())
            dist_per_joint.append(np.array(e_jt))


        per_joint_error = np.array(dist_per_joint).mean(axis=0).tolist()
        joint_names = MPII_NAMES

        logger.info('=== JOINTS ===')
        for idx in range(len(joint_names)):
            logger.info('%s : %s' % (joint_names[idx], per_joint_error[idx]))

        results = {
            'hm36_17j': np.asarray(dist).mean(),
            'hm36_17j_align': np.array(dist_align).mean(),
            'hm36_17j_14': np.asarray(dist_14).mean(),
            'hm36_17j_14_al': np.array(dist_14_align).mean(),
            'hm36_17j_x': np.array(dist_x).mean(),
            'hm36_17j_y': np.array(dist_y).mean(),
            'hm36_17j_z': np.array(dist_z).mean(),
        }

        logger.info('=== RESULTS ===')
        for k, v in results.items():
            logger.info('%s : %s' % (k, np.array(v).mean()))
        logger.info('===============')

        return np.asarray(dist).mean()