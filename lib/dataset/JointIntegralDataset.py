import logging
import numpy as np
from torch.utils.data import Dataset

from lib.core.integral_loss import get_label_func

from lib.utils.augmentation import load_occluders

H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Thorax'
H36M_NAMES[9] = 'Neck/Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Thorax'
MPII_NAMES[8]  = 'Neck/Nose'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'

logger = logging.getLogger(__name__)

H36M_TO_MPII_PERM = np.array([H36M_NAMES.index(h) for h in MPII_NAMES if h != '' and h in H36M_NAMES])


class JointsIntegralDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train):
        self.cfg = cfg
        self.is_train = is_train

        self.root = root
        self.image_set = image_set

        self.is_train = is_train

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        self.rect_3d_width = 2000.
        self.rect_3d_height = 2000.

        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([58.395, 57.120, 57.375])
        self.num_cams = cfg.DATASET.NUM_CAMS

        self.label_func = get_label_func()

        self.occluders = load_occluders(cfg.DATASET.VOC) if cfg.DATASET.OCCLUSION and is_train else None

        self.cam_config = []
        self.parent_ids = None
        self.db_length = 0

        self.db = []


    def __len__(self, ):
        return self.db_length


    def __getitem__(self, idx):
        raise NotImplementedError


    def evaluate(self, preds, save_path=None, debug=False):
        raise NotImplementedError


