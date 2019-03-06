import logging
import os
import json
import numpy as np
from scipy.io import loadmat, savemat
import copy

from lib.utils.img_utils import get_single_patch_sample
from lib.utils.utils import calc_kpt_bound
from lib.dataset.JointIntegralDataset import JointsIntegralDataset

logger = logging.getLogger(__name__)

class MPIIDataset(JointsIntegralDataset):
    def __init__(self, cfg, root, image_set, is_train):
        super().__init__(cfg, root, image_set, is_train)


        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        the_db = copy.deepcopy(self.db[idx])

        img_patch, label, label_weight, _, _ = get_single_patch_sample(the_db['image'], the_db['center_x'],
                                                                       the_db['center_y'], the_db['width'],
                                                                       the_db['height'], the_db['joints_3d'].copy(),
                                                                       the_db['joints_3d_vis'].copy(),
                                                                       self.flip_pairs.copy(),
                                                                       self.parent_ids.copy(),
                                                                       self.patch_width, self.patch_height,
                                                                       self.rect_3d_width, self.rect_3d_height,
                                                                       self.mean, self.std, self.is_train, self.label_func,
                                                                       occluder=self.occluders, DEBUG=False)

        meta = {
            'image': the_db['image'],
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        aspect_ratio = self.patch_width * 1.0 / self.patch_height

        gt_db = []
        for a in anno:
            # joints and vis
            jts_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            jts_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                jts = np.array(a['joints'])
                jts[:, 0:2] = jts[:, 0:2] - 1
                jts_vis = np.array(a['joints_vis'])
                assert len(jts) == self.num_joints, 'joint num diff: {} vs {}'.format(len(jts), self.num_joints)
                jts_3d[:, 0:2] = jts[:, 0:2]
                jts_3d_vis[:, 0] = jts_vis[:]
                jts_3d_vis[:, 1] = jts_vis[:]

            if np.sum(jts_3d_vis[:, 0]) < 2:  # only one joint visible, skip
                continue

            u, d, l, r = calc_kpt_bound(jts_3d, jts_3d_vis)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x = center[0]
            c_y = center[1]
            assert c_x >= 1

            w = r - l
            h = d - u

            assert w > 0
            assert h > 0

            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio

            width = w * 1.25
            height = h * 1.25

            img_path = os.path.join(self.root, 'images', a['image'])
            gt_db.append({
                'image': img_path,
                'center_x': c_x,
                'center_y': c_y,
                'width': width,
                'height': height,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': jts_3d,
                'joints_3d_vis': jts_3d_vis,
            })

        return gt_db

    def evaluate(self, preds, save_path=None, debug=False):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if save_path:
            pred_file = os.path.join(save_path, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in self.cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(self.cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        # name_value = OrderedDict(name_value)

        return name_value, np.sum(PCKh * jnt_ratio)
