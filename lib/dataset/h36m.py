import torch
import pickle as pkl
import random
import os
import logging
import numpy as np
import copy

from lib.utils.prep_h36m import CamBackProj, compute_similarity_transform
from lib.dataset.JointIntegralDataset import JointsIntegralDataset, H36M_NAMES, MPII_NAMES
from lib.utils.data_utils import define_actions
from lib.utils.img_utils import get_single_patch_sample

logger = logging.getLogger(__name__)

H36M_TO_MPII_PERM = np.array([H36M_NAMES.index(h) for h in MPII_NAMES if h != '' and h in H36M_NAMES])


class H36M_Integral(JointsIntegralDataset):
    def __init__(self, cfg, root, image_set, is_train):
        super().__init__(cfg, root, image_set, is_train)

        self.parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 8, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int)

        self.cam_config = [[1, 2], [0, 3], [0, 3], [1, 2]] # Camera neighborhoods

        self.db = self._get_train_db() if is_train else self._get_val_db()

        logger.info('=> load {} samples'.format(self.db_length))


    def __getitem__(self, idx):
        if self.is_train and self.cfg.DATASET.TRI:
            # select a random camera from db
            cam_1 = np.random.randint(self.num_cams)
            # select a neighboring camera
            cam_2 = self.cam_config[cam_1][0] if random.random() <= 0.5 else self.cam_config[cam_1][1]

            # get db records for each view
            db_rec_1 = copy.deepcopy(self.db[cam_1][idx])
            db_rec_2 = copy.deepcopy(self.db[cam_2][idx])

            # there you go
            bundle_1 = self.get_data(db_rec_1)
            bundle_2 = self.get_data(db_rec_2)

            return {'cam_1': bundle_1, 'cam_2': bundle_2}
        else:
            db_rec = copy.deepcopy(self.db[idx])
            return self.get_data(db_rec)


    def get_data(self, the_db):

        image_file = os.path.join(self.root, the_db['image'])


        cam = the_db['cam']

        joints_vis = the_db['joints_3d_vis'].copy()
        joints_vis[:,2] *= self.cfg.DATASET.Z_WEIGHT

        img_patch, label, label_weight, scale, rot = get_single_patch_sample(image_file, the_db['center_x'],
                                                                 the_db['center_y'], the_db['width'],
                                                                 the_db['height'], the_db['joints_3d'].copy(),
                                                                 joints_vis,
                                                                 the_db['flip_pairs'].copy(), the_db['parent_ids'].copy(),
                                                                 self.patch_width, self.patch_height,
                                                                 self.rect_3d_width, self.rect_3d_height,
                                                                 self.mean, self.std, self.is_train, self.label_func,
                                                                 occluder=self.occluders, DEBUG=self.cfg.DEBUG.DEBUG)

        meta = {
            'image'   : image_file,
            'center_x': the_db['center_x'],
            'center_y': the_db['center_y'],
            'width'   : the_db['width'],
            'height'  : the_db['height'],
            'scale'   : float(scale),
            'rot'     : float(rot),
            'R'       : cam.R,
            'T'       : cam.T,
            'f'       : cam.f,
            'c'       : cam.c,
            'projection_matrix': cam.projection_matrix
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta


    def _get_train_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set + '.pkl')

        with open(file_name, 'rb') as anno_file:
            anno = pkl.load(anno_file)

        if isinstance(anno, dict):
            gt_db = [[] for i in range(self.num_cams)]  # for each cameras construct a database
            rnd_subset = np.random.permutation(len(anno[1]))
            for idx in rnd_subset:
                for cid in range(self.num_cams):
                    a = anno[cid+1][idx]
                    gt_db[cid].append(a)

            self.db_length = len(gt_db[0])

            if not self.cfg.DATASET.TRI:
                temp_db = []
                for db in gt_db:
                    temp_db += db
                gt_db = temp_db

                # Shuffle the dataset
                random.shuffle(gt_db)

                self.db_length = len(gt_db)
        else:
            gt_db = []

            for idx in range(len(anno)):
                a = anno[idx]
                gt_db.append(a)

            # Shuffle the dataset
            random.shuffle(gt_db)
            self.db_length = len(gt_db)

        return gt_db


    def _get_val_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set + '.pkl')

        with open(file_name, 'rb') as anno_file:
            anno = pkl.load(anno_file)

        if isinstance(anno, dict):
            gt_db = [[] for i in range(self.num_cams)]  # for each cameras construct a database
            rnd_subset = np.random.permutation(len(anno[1]))
            for idx in rnd_subset:
                for cid in range(self.num_cams):
                    a = anno[cid + 1][idx]
                    gt_db[cid].append(a)

            temp_db = []
            for db in gt_db:
                temp_db += db
            gt_db = temp_db

        else:
            gt_db = []

            for idx in range(len(anno)):
                a = anno[idx]
                gt_db.append(a)

        self.db_length = len(gt_db)

        return gt_db


    def evaluate(self, preds, save_path=None, debug=False, actionwise=False):
        preds = preds[:, :, 0:3]

        gt_poses = []
        pred_poses = []

        gt_poses_glob = []
        pred_poses_glob = []
        pred_2d_poses = []
        all_images = []

        gts = self.db

        sample_num = preds.shape[0]
        root = 6 if self.cfg.DATASET.MPII_ORDER else 0
        pred_to_save = []

        j14 = [0,1,2,3,4,5,6,7,10,11,12,13,14,15] if self.cfg.DATASET.MPII_ORDER else [0,1,2,4,5,6,7,8,9,10,11,12,14,15]

        dist = []
        dist_align = []
        dist_norm = []
        dist_14 = []
        dist_14_align = []
        dist_14_norm = []
        dist_x = []
        dist_y = []
        dist_z = []
        dist_per_joint = []
        pck = []

        if actionwise:
            acts = define_actions('All')
            dist_actions = {}
            dist_actions_align = {}
            for act in acts:
                dist_actions[act] = []
                dist_actions_align[act] = []

        for n_sample in range(0, sample_num):
            gt = gts[n_sample]
            # Org image info
            fl = gt['fl'][0:2]
            c_p = gt['c_p'][0:2]

            gt_3d_root = np.reshape(gt['pelvis'], (1, 3))
            gt_2d_kpt = gt['joints_3d'].copy()
            gt_vis = gt['joints_3d_vis'].copy()
            # subj = gt['subject']
            if actionwise:
                action = gt['action']

            # get camera depth from root joint
            pre_2d_kpt = preds[n_sample].copy()

            if self.cfg.DATASET.MPII_ORDER:
                gt_2d_kpt = gt_2d_kpt[H36M_TO_MPII_PERM,:]

            pre_2d_kpt[:, 2] = pre_2d_kpt[:, 2] + gt_3d_root[0, 2]
            gt_2d_kpt[:, 2] = gt_2d_kpt[:, 2] + gt_3d_root[0, 2]

            joint_num = pre_2d_kpt.shape[0]

            # back project
            pre_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
            gt_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)

            for n_jt in range(0, joint_num):
                pre_3d_kpt[n_jt, 0], pre_3d_kpt[n_jt, 1], pre_3d_kpt[n_jt, 2] = \
                    CamBackProj(pre_2d_kpt[n_jt, 0], pre_2d_kpt[n_jt, 1], pre_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
                gt_3d_kpt[n_jt, 0], gt_3d_kpt[n_jt, 1], gt_3d_kpt[n_jt, 2] = \
                    CamBackProj(gt_2d_kpt[n_jt, 0], gt_2d_kpt[n_jt, 1], gt_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])

            # align
            _, Z, T, b, c = compute_similarity_transform(gt_3d_kpt, pre_3d_kpt, compute_optimal_scale=True)
            pre_3d_kpt_align = (b * pre_3d_kpt.dot(T)) + c
            pre_3d_kpt_norm = b * pre_3d_kpt

            # should align root, required by protocol #1
            pre_3d_kpt = pre_3d_kpt - pre_3d_kpt[root]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[root]
            pre_3d_kpt_align = pre_3d_kpt_align - pre_3d_kpt_align[root]
            pre_3d_kpt_norm = pre_3d_kpt_norm - pre_3d_kpt_norm[root]

            if self.cfg.DATASET.MPII_ORDER:
                gt_poses.append(gt_3d_kpt)
                pred_poses.append(pre_3d_kpt)
            else:
                gt_poses.append(gt_3d_kpt[H36M_TO_MPII_PERM,:])
                pred_poses.append(pre_3d_kpt[H36M_TO_MPII_PERM,:])

            diff = (gt_3d_kpt - pre_3d_kpt)
            diff_align = (gt_3d_kpt - pre_3d_kpt_align)
            diff_norm = (gt_3d_kpt - pre_3d_kpt_norm)

            e_jt = []
            e_jt_align = []
            e_jt_norm = []
            e_jt_14 = []
            e_jt_14_align = []
            e_jt_14_norm = []
            e_jt_x = []
            e_jt_y = []
            e_jt_z = []

            for n_jt in range(0, joint_num):
                e_jt.append(np.linalg.norm(diff[n_jt]))
                e_jt_align.append(np.linalg.norm(diff_align[n_jt]))
                e_jt_norm.append(np.linalg.norm(diff_norm[n_jt]))
                e_jt_x.append(np.sqrt(diff[n_jt][0]**2))
                e_jt_y.append(np.sqrt(diff[n_jt][1]**2))
                e_jt_z.append(np.sqrt(diff[n_jt][2]**2))

                if np.linalg.norm(diff[n_jt]) >= 150:
                    pck.append(0)
                else:
                    pck.append(1)

            for jt in j14:
                e_jt_14.append(np.linalg.norm(diff[jt]))
                e_jt_14_align.append(np.linalg.norm(diff_align[jt]))
                e_jt_14_norm.append(np.linalg.norm(diff_norm[jt]))

            if self.cfg.DEBUG.DEBUG:
                cam = gt['cam']
                pred = cam.camera_to_world_frame(pre_3d_kpt)
                gt_pt = cam.camera_to_world_frame(gt_3d_kpt)

                gt_poses_glob.append(gt_pt)
                pred_poses_glob.append(pred)
                pred_2d_poses.append(pre_2d_kpt)
                all_images.append(self.root + gts[n_sample]['image'])
                import cv2
                import matplotlib.pyplot as plt
                from lib.utils.vis import drawskeleton, show3Dpose

                if self.cfg.DATASET.MPII_ORDER:
                    m = 1
                else:
                    m = 2

                img = cv2.imread(self.root + gts[n_sample]['image'])

                fig = plt.figure(figsize=(19.2, 10.8))

                lc = (255, 0, 0), '#ff0000'
                rc = (0, 0, 255), '#0000ff'
                ax = fig.add_subplot('131')
                drawskeleton(img, pre_2d_kpt, thickness=3, lcolor=lc[0], rcolor=rc[0], mpii=m)
                drawskeleton(img, gt_2d_kpt, thickness=3, lcolor=(255, 0, 0), rcolor=(255, 0, 0), mpii=m)
                ax.imshow(img[:,:,::-1])

                ax = fig.add_subplot('132', projection='3d', aspect=1)
                show3Dpose(gt_pt, ax, radius=750, lcolor=lc[1], rcolor=rc[1], mpii=m)

                ax = fig.add_subplot('133', projection='3d', aspect=1)
                show3Dpose(pred, ax, radius=750, lcolor=lc[1], rcolor=rc[1], mpii=m)
                ax.set_title(str(n_sample) + ' %s' % np.array(e_jt).mean())

                plt.show()

            dist.append(np.array(e_jt).mean())
            dist_align.append(np.array(e_jt_align).mean())
            dist_norm.append(np.array(e_jt_norm).mean())
            dist_14.append(np.array(e_jt_14).mean())
            dist_14_align.append(np.array(e_jt_14_align).mean())
            dist_14_norm.append(np.array(e_jt_14_norm).mean())
            dist_x.append(np.array(e_jt_x).mean())
            dist_y.append(np.array(e_jt_y).mean())
            dist_z.append(np.array(e_jt_z).mean())
            dist_per_joint.append(np.array(e_jt))

            if actionwise:
                dist_actions[action].append(np.array(e_jt).mean())
                dist_actions_align[action].append(np.array(e_jt_align).mean())

            pred_to_save.append({'pred': pre_3d_kpt,
                                 'align_pred': pre_3d_kpt_align,
                                 'gt': gt_3d_kpt})

        per_joint_error = np.array(dist_per_joint).mean(axis=0).tolist()
        joint_names = MPII_NAMES if self.cfg.DATASET.MPII_ORDER else H36M_NAMES

        for idx in range(len(joint_names)):
            print(joint_names[idx], per_joint_error[idx])
        if actionwise:
            print('========================')
            for k, v in dist_actions.items():
                print(k, np.array(v).mean())
            print('========================')

            print('========================')
            for k, v in dist_actions_align.items():
                print(k, np.array(v).mean())
            print('========================')

        name_value = [
            ('hm36_17j      :', np.asarray(dist).mean()),
            ('hm36_17j_align:', np.array(dist_align).mean()),
            ('hm36_17j_norm:', np.array(dist_norm).mean()),
            ('hm36_17j_14   :', np.asarray(dist_14).mean()),
            ('hm36_17j_14_al:', np.array(dist_14_align).mean()),
            ('hm36_17j_14_nm:', np.array(dist_14_norm).mean()),
            ('hm36_17j_x    :', np.array(dist_x).mean()),
            ('hm36_17j_y    :', np.array(dist_y).mean()),
            ('hm36_17j_z    :', np.array(dist_z).mean()),
        ]

        return name_value, np.asarray(dist).mean()