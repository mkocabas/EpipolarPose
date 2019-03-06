import numpy as np
import pickle


# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
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

# Stacked Hourglass produces 16 joints. These are the names.
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

H36M_TO_MPII_PERM = np.array([H36M_NAMES.index(h) for h in MPII_NAMES if h != '' and h in H36M_NAMES])

calc_jts = [0,1,2,3,4,5,6,7,10,11,12,13,14,15]

dims_to_use = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

s_hm36_2_mpii_jt = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]

DIM_TO_USE_3D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]

root_dir = '/media/muhammed/Other/RESEARCH/datasets/Human3.6M/ours'

ACTIONS_ALL = ["Directions","Discussion","Eating","Greeting",
             "Phoning","Photo","Posing","Purchases",
             "Sitting","SittingDown","Smoking","Waiting",
             "WalkDog","Walking","WalkTogether"]

TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

s_36_bone_jts = np.array([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                          [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])

s_36_flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)
s_36_parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 8, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int)

mpii_flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
mpii_parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

def save(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f, protocol=4)

def check_image(img):
    if img.shape == (1002, 1000, 3):
        img = img[:1000, :, :]
    elif img.shape == (1002, 1000):
        img = img[:1000, :]
    return img

def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v):
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth
    z = depth
    return x, y, z

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
    """

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def CamProj(x, y, z, fx, fy, u, v):
    cam_x = x / z * fx
    cam_x = cam_x + u
    cam_y = y / z * fy
    cam_y = cam_y + v
    return cam_x, cam_y

def from_worldjt_to_imagejt(joint_num, rot, keypoints, trans, fl, c_p, rect_3d_width, rect_3d_height, mpii=False):
    # project to image space
    pt_3d = np.zeros((joint_num, 3), dtype=np.float)
    pt_2d = np.zeros((joint_num, 3), dtype=np.float)

    root_joint = 6 if mpii else 0

    for n_jt in range(0, joint_num):

        pt_3d[n_jt] = np.dot(rot, keypoints[n_jt] - trans.reshape(3))
        pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pt_3d[n_jt, 0], pt_3d[n_jt, 1], pt_3d[n_jt, 2], fl[0], fl[1],
                                                 c_p[0], c_p[1])
        pt_2d[n_jt, 2] = pt_3d[n_jt, 2]

    pelvis3d = pt_3d[root_joint]
    # build 3D bounding box centered on pelvis, size 2000^2
    rect3d_lt = pelvis3d - [rect_3d_width / 2, rect_3d_height / 2, 0]
    rect3d_rb = pelvis3d + [rect_3d_width / 2, rect_3d_height / 2, 0]
    # back-project 3D BBox to 2D image
    rect2d_l, rect2d_t = CamProj(rect3d_lt[0], rect3d_lt[1], rect3d_lt[2], fl[0], fl[1], c_p[0], c_p[1])
    rect2d_r, rect2d_b = CamProj(rect3d_rb[0], rect3d_rb[1], rect3d_rb[2], fl[0], fl[1], c_p[0], c_p[1])

    # Subtract pelvis depth
    pt_2d[:, 2] = pt_2d[:, 2] - pelvis3d[2]
    pt_2d = pt_2d.reshape((joint_num, 3))
    vis = np.ones((joint_num, 3), dtype=np.float)

    return rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d


def save_annotations():
    # will be released after proper testing and cleaning
    raise NotImplementedError

def save_triangulations():
    # will be released after proper testing and cleaning
    raise NotImplementedError
